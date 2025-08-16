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
    from config import get_config, get_tts_config, get_audio_config, get_server_config, get_websocket_config
    config = get_config()
    tts_config = get_tts_config()
    audio_config = get_audio_config()
    server_config = get_server_config()
    websocket_config = get_websocket_config()
except ImportError:
    # Fallback configuration if config.py is not available
    config = {
        "tts": {"speech_rate": 150, "volume": 0.9, "voice_id": None},
        "audio": {"sample_rate": 44100, "bit_depth": 16, "channels": 1, "chunk_size": 50, "min_text_length": 10},
        "server": {"host": "0.0.0.0", "port": 8000, "debug": False, "log_level": "INFO"},
        "websocket": {
            "bidirectional_streaming": {"enabled": True, "max_concurrent_tasks": 10, "immediate_send": True},
            "latency_optimization": {"min_text_length": 5, "preemptive_generation": True}
        }
    }
    tts_config = config["tts"]
    audio_config = config["audio"]
    server_config = config["server"]
    websocket_config = config["websocket"]

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
        """Convert text to audio using Chatterbox TTS - outputs exact 44.1 kHz, 16-bit, mono PCM"""
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
            
            # Ensure exact 44.1 kHz, 16-bit, mono PCM format
            target_sample_rate = 44100  # Fixed as per requirements
            target_bit_depth = 16       # Fixed as per requirements
            target_channels = 1         # Fixed as per requirements (mono)
            
            # Resample to exactly 44.1 kHz if needed
            if hasattr(self.model, 'sr') and self.model.sr != target_sample_rate:
                wav = librosa.resample(wav, orig_sr=self.model.sr, target_sr=target_sample_rate)
            
            # Ensure mono (single channel)
            if wav.ndim > 1 and wav.shape[1] > 1:
                wav = wav.mean(axis=1)  # Convert stereo to mono by averaging
            
            # Convert to exact 16-bit PCM
            wav = (wav * 32767).astype(np.int16)
            
            # Convert to bytes
            return wav.tobytes()
                
        except Exception as e:
            logger.error(f"Chatterbox TTS generation failed: {e}")
            return None
    

    
    def generate_character_alignments(self, text: str, audio_duration_ms: float) -> Dict:
        """Generate character alignment data matching the required format"""
        if not text.strip():
            return {"chars": [], "char_start_times_ms": [], "char_durations_ms": []}
        
        # Include all characters including punctuation and whitespace
        chars = list(text)
        total_chars = len(chars)
        
        if total_chars == 0:
            return {"chars": [], "char_start_times_ms": [], "char_durations_ms": []}
        
        # More sophisticated alignment algorithm
        char_start_times = []
        char_durations = []
        
        # Base timing parameters
        base_char_duration = 70  # Base duration per character in ms
        punctuation_multiplier = 1.5  # Punctuation takes longer
        space_multiplier = 0.8  # Spaces are shorter
        word_boundary_multiplier = 1.2  # Word boundaries take longer
        
        current_time = 0
        
        for i, char in enumerate(chars):
            # Calculate character duration based on type
            if char.isspace():
                char_duration = int(base_char_duration * space_multiplier)
            elif char in '.,!?;:':
                char_duration = int(base_char_duration * punctuation_multiplier)
            elif i > 0 and (chars[i-1].isspace() or chars[i-1] in '.,!?;:'):
                # Word boundary
                char_duration = int(base_char_duration * word_boundary_multiplier)
            else:
                char_duration = base_char_duration
            
            # Add some natural variation
            variation = int(char_duration * 0.1 * (hash(char + str(i)) % 3 - 1))
            char_duration = max(30, char_duration + variation)  # Minimum 30ms
            
            char_start_times.append(int(current_time))
            char_durations.append(char_duration)
            
            current_time += char_duration
        
        # Scale timing to match actual audio duration
        if current_time > 0:
            scale_factor = audio_duration_ms / current_time
            char_start_times = [int(time * scale_factor) for time in char_start_times]
            char_durations = [int(duration * scale_factor) for duration in char_durations]
        
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
        self.connection_states: Dict[str, bool] = {}  # Track connection state
        self.connection_tasks: Dict[str, asyncio.Task] = {}  # Track running tasks
        self.audio_queues: Dict[str, asyncio.Queue] = {}  # Queues for audio chunks
    
    async def connect(self, websocket: WebSocket, connection_id: str):
        await websocket.accept()
        self.active_connections[connection_id] = websocket
        self.connection_buffers[connection_id] = ""
        self.connection_states[connection_id] = True
        self.audio_queues[connection_id] = asyncio.Queue()
        logger.info(f"WebSocket connected: {connection_id}")
    
    def disconnect(self, connection_id: str):
        if connection_id in self.active_connections:
            del self.active_connections[connection_id]
        if connection_id in self.connection_buffers:
            del self.connection_buffers[connection_id]
        if connection_id in self.connection_states:
            self.connection_states[connection_id] = False
            del self.connection_states[connection_id]
        if connection_id in self.audio_queues:
            del self.audio_queues[connection_id]
        if connection_id in self.connection_tasks:
            # Cancel any running tasks
            task = self.connection_tasks[connection_id]
            if not task.done():
                task.cancel()
            del self.connection_tasks[connection_id]
        logger.info(f"WebSocket disconnected: {connection_id}")
    
    def is_connected(self, connection_id: str) -> bool:
        """Check if a connection is still active"""
        if (connection_id not in self.active_connections or 
            connection_id not in self.connection_states or 
            not self.connection_states[connection_id]):
            return False
        
        # Additional check: verify the WebSocket object is still valid
        websocket = self.active_connections[connection_id]
        try:
            # Try to access a property to check if WebSocket is still valid
            _ = websocket.client_state
            return True
        except Exception:
            # WebSocket is no longer valid, mark as disconnected
            logger.warning(f"WebSocket {connection_id} is no longer valid, marking as disconnected")
            self.connection_states[connection_id] = False
            return False
    
    async def safe_send_text(self, websocket: WebSocket, connection_id: str, text: str) -> bool:
        """Safely send text to WebSocket, returns True if successful"""
        if not self.is_connected(connection_id):
            return False
        
        try:
            await websocket.send_text(text)
            return True
        except Exception as e:
            logger.warning(f"Failed to send message to {connection_id}: {e}")
            self.disconnect(connection_id)
            return False
    
    async def process_message(self, websocket: WebSocket, connection_id: str, message: dict):
        """Process incoming WebSocket message according to specification"""
        try:
            # Check if connection is still valid
            if not self.is_connected(connection_id):
                logger.warning(f"Processing message for disconnected connection: {connection_id}")
                return
            
            text = message.get("text", "")
            flush = message.get("flush", False)
            
            # Handle initial space character (first chunk from client)
            if text == " " and not self.connection_buffers[connection_id]:
                logger.info(f"Received initial space character from {connection_id}")
                return  # Don't add space to buffer, just acknowledge
            
            # Handle connection close (empty string without flush)
            if text == "" and not flush:
                logger.info(f"Client requested connection close for {connection_id}")
                # Don't close immediately - wait for any pending audio generation to complete
                # The connection will be closed by the client after all processing is done
                return
            
            # Add text to buffer (skip initial space)
            if text != " ":
                self.connection_buffers[connection_id] += text
            
            # Get optimized settings from config
            min_length = websocket_config.get("latency_optimization", {}).get("min_text_length", 5)
            preemptive = websocket_config.get("latency_optimization", {}).get("preemptive_generation", True)
            
            # Generate audio if we have text and either flush is True or we have substantial text
            buffer_text = self.connection_buffers[connection_id].strip()
            
            if buffer_text and (flush or len(buffer_text) > min_length):
                # Start audio generation as a concurrent task for low latency
                max_tasks = websocket_config.get("bidirectional_streaming", {}).get("max_concurrent_tasks", 10)
                
                # Check if we're not exceeding max concurrent tasks
                active_tasks = [task for task in asyncio.all_tasks() if not task.done()]
                if len(active_tasks) < max_tasks:
                    asyncio.create_task(self.generate_and_send_audio_async(websocket, connection_id, buffer_text))
                else:
                    # If too many tasks, process synchronously
                    await self.generate_and_send_audio_async(websocket, connection_id, buffer_text)
                
                # Clear buffer after processing (unless flush is True)
                if not flush:
                    self.connection_buffers[connection_id] = ""
            
            # Preemptive generation for better latency
            elif preemptive and buffer_text and len(buffer_text) > 2:
                # Start generating audio for partial text to reduce latency
                asyncio.create_task(self.generate_and_send_audio_async(websocket, connection_id, buffer_text))
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            if self.is_connected(connection_id):
                await self.safe_send_text(websocket, connection_id, json.dumps({"error": str(e)}))
    
    async def generate_and_send_audio_async(self, websocket: WebSocket, connection_id: str, text: str):
        """Generate audio for text and send via WebSocket - outputs exact format as per requirements"""
        try:
            # Check if connection is still valid before processing
            if not self.is_connected(connection_id):
                logger.warning(f"Generating audio for disconnected connection: {connection_id}")
                return
            
            # Generate audio
            audio_data = tts_manager.text_to_audio(text)
            
            if audio_data:
                # Convert to Base64
                audio_base64 = base64.b64encode(audio_data).decode('utf-8')
                
                # Calculate audio duration for 44.1 kHz, 16-bit, mono PCM
                # Formula: duration_ms = (bytes / (sample_rate * channels * bytes_per_sample)) * 1000
                sample_rate = 44100  # Fixed as per requirements
                channels = 1         # Fixed as per requirements (mono)
                bytes_per_sample = 2  # 16-bit = 2 bytes
                
                sample_count = len(audio_data) // (channels * bytes_per_sample)
                duration_ms = (sample_count / sample_rate) * 1000
                
                # Generate character alignments
                alignments = tts_manager.generate_character_alignments(text, duration_ms)
                
                # Send audio chunk in exact required format
                response = {
                    "audio": audio_base64,
                    "alignment": alignments
                }
                
                if self.is_connected(connection_id):
                    success = await self.safe_send_text(websocket, connection_id, json.dumps(response))
                    if success:
                        logger.info(f"Sent audio chunk for text: '{text[:50]}...' ({len(audio_data)} bytes, {duration_ms:.1f}ms)")
                    else:
                        logger.warning(f"Failed to send audio to {connection_id}")
                else:
                    logger.warning(f"Connection {connection_id} disconnected before audio could be sent")
            else:
                logger.warning(f"Failed to generate audio for text: '{text}'")
                
        except Exception as e:
            logger.error(f"Error generating audio: {e}")
            if self.is_connected(connection_id):
                await self.safe_send_text(websocket, connection_id, json.dumps({"error": f"Audio generation failed: {str(e)}"}))
    
    async def start_bidirectional_streaming(self, websocket: WebSocket, connection_id: str):
        """Start bidirectional streaming with concurrent send/receive"""
        try:
            # Create separate tasks for receiving and processing
            receive_task = asyncio.create_task(self.receive_messages(websocket, connection_id))
            process_task = asyncio.create_task(self.process_audio_queue(websocket, connection_id))
            
            # Store tasks for cleanup
            self.connection_tasks[connection_id] = asyncio.gather(receive_task, process_task)
            
            # Wait for either task to complete (indicating disconnection)
            await self.connection_tasks[connection_id]
            
        except asyncio.CancelledError:
            logger.info(f"Bidirectional streaming cancelled for {connection_id}")
        except Exception as e:
            logger.error(f"Error in bidirectional streaming: {e}")
        finally:
            # Cancel any remaining tasks
            if connection_id in self.connection_tasks:
                task = self.connection_tasks[connection_id]
                if not task.done():
                    task.cancel()
    
    async def receive_messages(self, websocket: WebSocket, connection_id: str):
        """Receive messages from client"""
        try:
            while self.is_connected(connection_id):
                data = await websocket.receive_text()
                message = json.loads(data)
                await self.process_message(websocket, connection_id, message)
        except WebSocketDisconnect:
            logger.info(f"WebSocket disconnected by client: {connection_id}")
        except Exception as e:
            logger.error(f"Error receiving messages: {e}")
    
    async def process_audio_queue(self, websocket: WebSocket, connection_id: str):
        """Process audio queue for sending (if needed for queued operations)"""
        try:
            while self.is_connected(connection_id):
                # This can be used for queued audio operations if needed
                # For now, we send audio immediately in generate_and_send_audio_async
                await asyncio.sleep(0.01)  # Small delay to prevent busy waiting
        except Exception as e:
            logger.error(f"Error processing audio queue: {e}")
    
    # Keep the old method for backward compatibility
    async def generate_and_send_audio(self, websocket: WebSocket, connection_id: str, text: str):
        """Legacy method - now calls the async version"""
        await self.generate_and_send_audio_async(websocket, connection_id, text)

# Global WebSocket manager
websocket_manager = WebSocketManager()

@app.websocket("/ws/tts")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for TTS streaming with bidirectional concurrent operations"""
    connection_id = str(uuid.uuid4())
    
    try:
        await websocket_manager.connect(websocket, connection_id)
        
        # Send initial connection confirmation using safe send
        success = await websocket_manager.safe_send_text(websocket, connection_id, json.dumps({
            "status": "connected",
            "message": "TTS WebSocket connected successfully"
        }))
        
        if not success:
            logger.warning(f"Failed to send initial connection message to {connection_id}")
            return
        
        # Start bidirectional streaming with concurrent send/receive
        logger.info(f"Starting bidirectional streaming for connection: {connection_id}")
        await websocket_manager.start_bidirectional_streaming(websocket, connection_id)
            
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {connection_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        # Always clean up the connection
        websocket_manager.disconnect(connection_id)
        try:
            await websocket.close()
        except:
            pass  # WebSocket might already be closed

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
