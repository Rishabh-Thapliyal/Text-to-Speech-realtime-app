import asyncio
import base64
import json
import logging
import time
import warnings
from typing import Dict, List, Optional
import io
import wave
import numpy as np
import sys
import os
import tempfile

# Suppress specific deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*LoRACompatibleLinear.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.backends.cuda.sdp_kernel.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*past_key_values.*")

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import torch
import torchaudio as ta
import soundfile as sf
import librosa
import threading
from queue import Queue
import uuid

# Import configuration
try:
    from config import get_config, get_tts_config, get_audio_config, get_server_config
    config = get_config()
    tts_config = get_tts_config()
    audio_config = get_audio_config()
    server_config = get_server_config()
except ImportError:
    # Fallback configuration if config.py is not available
    config = {
        "tts": {"speech_rate": 150, "volume": 0.9, "voice_id": None},
        "audio": {"sample_rate": 44100, "bit_depth": 16, "channels": 1, "chunk_size": 50, "min_text_length": 10},
        "server": {"host": "0.0.0.0", "port": 8000, "debug": False, "log_level": "INFO"}
    }
    tts_config = config["tts"]
    audio_config = config["audio"]
    server_config = config["server"]

# Configure logging
log_level = getattr(logging, server_config.get("log_level", "INFO"))
logging.basicConfig(level=log_level)
logger = logging.getLogger(__name__)

app = FastAPI(title="SigIQ TTS WebSocket API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TTSManager:
    def __init__(self):
        self.model = None
        self.device = None
        self._init_engine()
    
    def _init_engine(self):
        """Initialize the Chatterbox TTS engine"""
        try:
            # Set device (GPU if available, else CPU)
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {self.device}")
            
            # Import and initialize Chatterbox TTS
            try:
                from chatterbox.tts import ChatterboxTTS
                
                logger.info("Loading Chatterbox TTS model...")
                self.model = ChatterboxTTS.from_pretrained(device=self.device)
                logger.info("Chatterbox TTS engine initialized successfully")
                
            except ImportError:
                logger.error("Chatterbox TTS not available. Please install with: pip install chatterbox-tts")
                raise
                
        except Exception as e:
            logger.error(f"Failed to initialize TTS engine: {e}")
            raise
    
    def text_to_audio(self, text: str) -> Optional[bytes]:
        """Convert text to audio using Chatterbox TTS"""
        if not text.strip():
            return None
        
        try:
            # Generate audio using Chatterbox
            wav = self.model.generate(text)
            
            # Convert to numpy array if it's a tensor
            if hasattr(wav, 'cpu'):
                wav = wav.cpu().numpy()
            
            # Ensure wav is 1D
            if wav.ndim > 1:
                wav = wav.squeeze()
            
            # Normalize audio
            wav = librosa.util.normalize(wav)
            
            # Convert to target format
            target_sample_rate = audio_config.get("sample_rate", 44100)
            target_bit_depth = audio_config.get("bit_depth", 16)
            
            # Resample if needed
            if hasattr(self.model, 'sr') and self.model.sr != target_sample_rate:
                wav = librosa.resample(wav, orig_sr=self.model.sr, target_sr=target_sample_rate)
            
            # Convert to target bit depth
            if target_bit_depth == 16:
                wav = (wav * 32767).astype(np.int16)
            elif target_bit_depth == 8:
                wav = (wav * 127).astype(np.int8)
            else:
                wav = (wav * 32767).astype(np.int16)
            
            # Convert to bytes
            return wav.tobytes()
                
        except Exception as e:
            logger.error(f"Chatterbox TTS generation failed: {e}")
            return None
    

    
    def generate_character_alignments(self, text: str, audio_duration_ms: float) -> Dict:
        """Generate character alignment data for the given text and audio duration"""
        if not text.strip():
            return {"chars": [], "char_start_times_ms": [], "char_durations_ms": []}
        
        chars = list(text)
        total_chars = len(chars)
        
        if total_chars == 0:
            return {"chars": [], "char_start_times_ms": [], "char_durations_ms": []}
        
        # Simple alignment algorithm - distribute time evenly among characters
        # In a production system, you'd use a more sophisticated alignment model
        char_duration = audio_duration_ms / total_chars
        
        char_start_times = []
        char_durations = []
        
        for i in range(total_chars):
            start_time = i * char_duration
            char_start_times.append(int(start_time))
            char_durations.append(int(char_duration))
        
        return {
            "chars": chars,
            "char_start_times_ms": char_start_times,
            "char_durations_ms": char_durations
        }

# Global TTS manager instance
tts_manager = TTSManager()

class WebSocketManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.connection_buffers: Dict[str, str] = {}
    
    async def connect(self, websocket: WebSocket, connection_id: str):
        await websocket.accept()
        self.active_connections[connection_id] = websocket
        self.connection_buffers[connection_id] = ""
        logger.info(f"WebSocket connected: {connection_id}")
    
    def disconnect(self, connection_id: str):
        if connection_id in self.active_connections:
            del self.active_connections[connection_id]
        if connection_id in self.connection_buffers:
            del self.connection_buffers[connection_id]
        logger.info(f"WebSocket disconnected: {connection_id}")
    
    async def process_message(self, websocket: WebSocket, connection_id: str, message: dict):
        """Process incoming WebSocket message"""
        try:
            text = message.get("text", "")
            flush = message.get("flush", False)
            
            # Handle connection close
            if text == "" and not flush:
                await websocket.close()
                self.disconnect(connection_id)
                return
            
            # Add text to buffer
            if text != " " or self.connection_buffers[connection_id]:  # Skip initial space
                self.connection_buffers[connection_id] += text
            
            # Generate audio if we have text and either flush is True or we have substantial text
            buffer_text = self.connection_buffers[connection_id].strip()
            min_length = audio_config.get("min_text_length", 10)
            if buffer_text and (flush or len(buffer_text) > min_length):
                await self.generate_and_send_audio(websocket, connection_id, buffer_text)
                
                # Clear buffer after processing (unless flush is True)
                if not flush:
                    self.connection_buffers[connection_id] = ""
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            await websocket.send_text(json.dumps({"error": str(e)}))
    
    async def generate_and_send_audio(self, websocket: WebSocket, connection_id: str, text: str):
        """Generate audio for text and send via WebSocket"""
        try:
            # Generate audio
            audio_data = tts_manager.text_to_audio(text)
            
            if audio_data:
                # Convert to Base64
                audio_base64 = base64.b64encode(audio_data).decode('utf-8')
                
                # Estimate audio duration (rough calculation based on PCM data)
                target_sample_rate = audio_config.get("sample_rate", 44100)
                target_channels = audio_config.get("channels", 1)
                target_bit_depth = audio_config.get("bit_depth", 16)
                bytes_per_sample = target_bit_depth // 8
                sample_count = len(audio_data) // (bytes_per_sample * target_channels)
                duration_ms = (sample_count / target_sample_rate) * 1000
                
                # Generate character alignments
                alignments = tts_manager.generate_character_alignments(text, duration_ms)
                
                # Send audio chunk
                response = {
                    "audio": audio_base64,
                    "alignment": alignments
                }
                
                await websocket.send_text(json.dumps(response))
                logger.info(f"Sent audio chunk for text: '{text[:50]}...' ({len(audio_data)} bytes)")
            else:
                logger.warning(f"Failed to generate audio for text: '{text}'")
                
        except Exception as e:
            logger.error(f"Error generating audio: {e}")
            await websocket.send_text(json.dumps({"error": f"Audio generation failed: {str(e)}"}))

# Global WebSocket manager
websocket_manager = WebSocketManager()

@app.websocket("/ws/tts")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for TTS streaming"""
    connection_id = str(uuid.uuid4())
    
    try:
        await websocket_manager.connect(websocket, connection_id)
        
        # Send initial connection confirmation
        await websocket.send_text(json.dumps({
            "status": "connected",
            "message": "TTS WebSocket connected successfully"
        }))
        
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Process the message
            await websocket_manager.process_message(websocket, connection_id, message)
            
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {connection_id}")
        websocket_manager.disconnect(connection_id)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        websocket_manager.disconnect(connection_id)
        try:
            await websocket.close()
        except:
            pass

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "SigIQ TTS WebSocket API",
        "endpoints": {
            "websocket": "/ws/tts",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "tts_engine": "available" if tts_manager.model else "unavailable",
        "active_connections": len(websocket_manager.active_connections)
    }

if __name__ == "__main__":
    import uvicorn
    
    # Use configuration values
    host = server_config.get("host", "0.0.0.0")
    port = server_config.get("port", 8000)
    debug = server_config.get("debug", False)
    
    print(f"🚀 Starting SigIQ TTS WebSocket Server")
    print(f"📍 Host: {host}")
    print(f"🔌 Port: {port}")
    print(f"🐛 Debug: {debug}")
    print(f"🎤 TTS Rate: {tts_config.get('speech_rate', 150)} WPM")
    print(f"🔊 Audio: {audio_config.get('sample_rate', 44100)}Hz, {audio_config.get('bit_depth', 16)}bit, {audio_config.get('channels', 1)}ch")
    print("=" * 50)
    
    uvicorn.run(
        app, 
        host=host, 
        port=port,
        log_level=server_config.get("log_level", "info").lower()
    )
