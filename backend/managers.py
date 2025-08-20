import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import asyncio
import base64
import json
import logging
from typing import Dict, List, Optional
import numpy as np
import asyncio
import torch
import librosa


# Import your config and logger as needed
from config import get_tts_config, get_audio_config, get_server_config, get_websocket_config


tts_config = get_tts_config()
audio_config = get_audio_config()
server_config = get_server_config()
websocket_config = get_websocket_config()

logger = logging.getLogger(__name__)

# Paste your TTSManager and WebSocketManager classes here

class TTSManager:
    def __init__(self):
        self.model = None
        self.device = None
        self.model_type = None
        self._init_engine()
    
    def _init_engine(self):
        """Initialize the selected TTS engine (Kokoro or Chatterbox)"""
        try:
            # Set device (GPU if available, else CPU)
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {self.device}")
            
            # Get selected model from configuration
            selected_model = tts_config.get("selected_model", "chatterbox")
            logger.info(f"Initializing TTS engine for model: {selected_model}")
            self.model_type = selected_model
            
            if selected_model == "kokoro":
                self._init_kokoro()
            elif selected_model == "chatterbox":
                self._init_chatterbox()
            else:
                raise ValueError(f"Unsupported model type: {selected_model}")
            
            # Verify model initialization
            if self.model is not None:
                logger.info(f"TTS engine initialized successfully: {self.model_type} ({type(self.model)})")
            else:
                raise RuntimeError(f"Model initialization failed for {selected_model}")
                
        except Exception as e:
            logger.error(f"Failed to initialize TTS engine: {e}")
            raise
    
    def _init_kokoro(self):
        """Initialize the Kokoro TTS engine with HiFi-GAN vocoder for better quality"""
        try:
            logger.info("Attempting to import Kokoro...")
            from kokoro import KPipeline
            logger.info("Kokoro import successful")
            
            logger.info("Loading Kokoro TTS model...")
            lang_code = tts_config.get("kokoro", {}).get("lang_code", "a")
            vocoder = tts_config.get("kokoro", {}).get("vocoder", "hifigan")
            logger.info(f"Using language code: {lang_code}, vocoder: {vocoder}")
            
            # Create KPipeline instance (vocoder selection handled internally by Kokoro)
            try:
                # Kokoro automatically selects the best available vocoder
                self.model = KPipeline(lang_code=lang_code)
                logger.info("Kokoro model created successfully")
                
                # Log vocoder information if available
                if hasattr(self.model, 'vocoder'):
                    logger.info(f"Kokoro TTS engine initialized with vocoder: {self.model.vocoder}")
                elif hasattr(self.model, 'model'):
                    if hasattr(self.model.model, 'vocoder'):
                        logger.info(f"Kokoro TTS engine initialized with vocoder: {self.model.model.vocoder}")
                    else:
                        logger.info("Kokoro TTS engine initialized (vocoder info not available)")
                else:
                    logger.info("Kokoro TTS engine initialized (vocoder selection handled internally)")
                    
            except Exception as e:
                logger.error(f"Failed to create Kokoro model: {e}")
                raise
            
            logger.info(f"Kokoro model created: {type(self.model)}")
            
            # Verify the model is callable
            if hasattr(self.model, '__call__'):
                logger.info("Kokoro model is callable")
            else:
                logger.warning("Kokoro model may not be callable")
            
            logger.info("Kokoro TTS engine initialized successfully")
            
        except ImportError as e:
            logger.error(f"Kokoro TTS not available: {e}")
            logger.error("Please install with: pip install kokoro>=0.9.2")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Kokoro TTS: {e}")
            logger.error(f"Error type: {type(e)}")
            raise
    
    def _init_chatterbox(self):
        """Initialize the Chatterbox TTS engine with enhanced quality settings"""
        try:
            from chatterbox.tts import ChatterboxTTS
            
            # Get Chatterbox configuration
            chatterbox_config = tts_config.get("chatterbox", {})
            vocoder = chatterbox_config.get("vocoder", "hifigan")
            
            logger.info(f"Loading Chatterbox TTS model (vocoder preference: {vocoder})...")
            
            # Initialize Chatterbox TTS (vocoder selection handled internally by the library)
            self.model = ChatterboxTTS.from_pretrained(device=self.device)
            
            # Log vocoder information if available
            if hasattr(self.model, 'vocoder'):
                logger.info(f"Chatterbox TTS engine initialized with vocoder: {self.model.vocoder}")
            elif hasattr(self.model, 'model'):
                # Check if vocoder info is in the model attribute
                if hasattr(self.model.model, 'vocoder'):
                    logger.info(f"Chatterbox TTS engine initialized with vocoder: {self.model.model.vocoder}")
                else:
                    logger.info("Chatterbox TTS engine initialized (vocoder info not available)")
            else:
                logger.info("Chatterbox TTS engine initialized successfully")
            
            # Note: HiFi-GAN vocoder selection is handled internally by ChatterboxTTS
            # The library automatically selects the best available vocoder
            
        except ImportError:
            logger.error("Chatterbox TTS not available. Please install with: pip install chatterbox-tts")
            raise
    
    def text_to_audio(self, text: str) -> Optional[bytes]:
        """Convert text to audio using the selected TTS engine - outputs exact 44.1 kHz, 16-bit, mono PCM"""
        if not text.strip():
            return None
        
        # Validate model state
        if not self.model or not self.model_type:
            logger.error(f"Invalid model state: model_type={self.model_type}, model={type(self.model) if self.model else 'None'}")
            
            # Try to recover by reinitializing the engine
            logger.info("Attempting to recover by reinitializing engine...")
            try:
                self._init_engine()
                if self.model and self.model_type:
                    logger.info(f"Recovery successful. Model: {self.model_type}")
                else:
                    logger.error("Recovery failed. Cannot process audio.")
                    return None
            except Exception as e:
                logger.error(f"Recovery failed: {e}")
                return None
        
        try:
            logger.debug(f"Processing text with {self.model_type} model (type: {type(self.model)})")
            
            if self.model_type == "kokoro":
                return self._kokoro_text_to_audio(text)
            elif self.model_type == "chatterbox":
                return self._chatterbox_text_to_audio(text)
            else:
                logger.error(f"Unsupported model type: {self.model_type}")
                return None
                
        except Exception as e:
            logger.error(f"{self.model_type.capitalize()} TTS generation failed: {e}")
            logger.error(f"Model type: {self.model_type}, Model object: {type(self.model)}")
            return None
    
    def _kokoro_text_to_audio(self, text: str) -> Optional[bytes]:
        """Convert text to audio using Kokoro TTS - High Quality Implementation"""
        try:
            # Get Kokoro configuration
            kokoro_config = tts_config.get("kokoro", {})
            voice = kokoro_config.get("voice", "af_heart")
            native_sample_rate = kokoro_config.get("sample_rate", 24000)
            
            logger.info(f"Generating Kokoro audio with voice: {voice}, sample_rate: {native_sample_rate}")
            
            # Generate audio using Kokoro (exactly like the original code)
            generator = self.model(text, voice=voice)
            
            # Collect all audio chunks (preserve original quality)
            audio_chunks = []
            for i, (gs, ps, audio) in enumerate(generator):
                audio_chunks.append(audio)
                logger.debug(f"Generated chunk {i}: gs={gs}, ps={ps}, audio_shape={audio.shape if hasattr(audio, 'shape') else 'scalar'}")
            
            if not audio_chunks:
                logger.warning("No audio chunks generated")
                return None
            
            # Concatenate all audio chunks
            wav = np.concatenate(audio_chunks)
            logger.info(f"Combined audio shape: {wav.shape}, dtype: {wav.dtype}")
            
            # Convert to numpy array if it's a tensor
            if hasattr(wav, 'cpu'):
                wav = wav.cpu().numpy()
                logger.debug("Converted tensor to numpy array")
            
            # Ensure wav is 1D
            if wav.ndim > 1:
                wav = wav.squeeze()
                logger.debug(f"Squeezed audio to shape: {wav.shape}")
            
            # HIGH QUALITY: Process to required format (44.1 kHz, 16-bit, mono PCM)
            return self._process_kokoro_audio_high_quality(wav, native_sample_rate)
            
        except Exception as e:
            logger.error(f"Kokoro TTS generation failed: {e}")
            logger.error(f"Error type: {type(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None

    def _process_kokoro_audio_high_quality(self, wav: np.ndarray, native_sample_rate: int) -> bytes:
        """High-quality audio processing for Kokoro to meet required format"""
        try:
            logger.info(f"Processing Kokoro audio: shape={wav.shape}, dtype={wav.dtype}, native_sr={native_sample_rate}")
            
            # Step 1: Preserve original dynamic range (don't over-normalize)
            max_val = np.max(np.abs(wav))
            if max_val > 1.0:
                # Only normalize if exceeding [-1, 1] range
                wav = wav / max_val
                logger.debug(f"Normalized audio to [-1, 1] range")
            
            # Step 2: High-quality resampling from 24kHz to 44.1kHz
            target_sample_rate = 44100
            if native_sample_rate != target_sample_rate:
                # Calculate new length for 44.1kHz
                new_length = int(len(wav) * target_sample_rate / native_sample_rate)
                
                # Use scipy for highest quality resampling
                try:
                    from scipy import signal
                    wav = signal.resample(wav, new_length)
                    logger.debug(f"Resampled from {native_sample_rate}Hz to {target_sample_rate}Hz using scipy")
                except ImportError:
                    # Fallback to librosa with high quality settings
                    wav = librosa.resample(wav, orig_sr=native_sample_rate, target_sr=target_sample_rate, 
                                         res_type='kaiser_best')  # Highest quality resampling
                    logger.debug(f"Resampled from {native_sample_rate}Hz to {target_sample_rate}Hz using librosa")
            
            # Step 3: Ensure mono (should already be mono, but double-check)
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
                logger.debug("Converted to mono")
            
            # Step 4: High-quality 16-bit PCM conversion
            # Use dithering to reduce quantization noise
            wav = self._apply_dithering(wav)
            
            # Convert to 16-bit PCM with proper scaling
            wav = (wav * 32767).astype(np.int16)
            logger.debug(f"Converted to 16-bit PCM, range: [{wav.min()}, {wav.max()}]")
            
            # Step 5: Convert to bytes
            audio_bytes = wav.tobytes()
            logger.info(f"Final audio: {len(audio_bytes)} bytes, {len(wav)} samples at {target_sample_rate}Hz")
            
            return audio_bytes
            
        except Exception as e:
            logger.error(f"High-quality Kokoro audio processing failed: {e}")
            return None

    def _apply_dithering(self, wav: np.ndarray) -> np.ndarray:
        """Apply dithering to reduce quantization noise during bit depth conversion - Shared by both Kokoro and Chatterbox"""
        try:
            # Generate triangular dither noise
            dither = np.random.triangular(-1, 0, 1, size=wav.shape) * 1e-6
            
            # Apply dither before quantization
            wav = wav + dither
            # Ensure we stay within [-1, 1] range
            wav = np.clip(wav, -1.0, 1.0)
            
            logger.debug("Applied dithering to reduce quantization noise")
            return wav
            
        except Exception as e:
            logger.warning(f"Dithering failed: {e}, continuing without dithering")
            return wav
    
    def _chatterbox_text_to_audio(self, text: str) -> Optional[bytes]:
        """Convert text to audio using Chatterbox TTS - High Quality Implementation"""
        try:
            logger.info("Generating Chatterbox audio...")
            
            # Generate audio using Chatterbox
            wav = self.model.generate(text)
            logger.debug(f"Raw Chatterbox audio shape: {wav.shape}, dtype: {wav.dtype}")
            
            # Convert to numpy array if it's a tensor
            if hasattr(wav, 'cpu'):
                wav = wav.cpu().numpy()
                logger.debug("Converted tensor to numpy array")
            
            # Ensure wav is 1D
            if wav.ndim > 1:
                wav = wav.squeeze()
                logger.debug(f"Squeezed audio to shape: {wav.shape}")
            
            # Get sample rate from model (Chatterbox typically uses 22.05kHz)
            native_sample_rate = getattr(self.model, 'sr', 22050) # 24000Hz is common for Chatterbox
            channels = 1 if wav.ndim == 1 else wav.shape[1]
            bit_depth = wav.dtype.itemsize * 8  # e.g., float32 = 32 bits, int16 = 16 bits
            logger.info(
                f"Chatterbox native sample rate: {native_sample_rate}Hz, "
                f"Channels: {channels}, Bit depth: {bit_depth}-bit, Dtype: {wav.dtype}"
            )

            # Save raw audio to WAV file in current working directory
            import os
            from scipy.io import wavfile
            save_path = os.path.join(os.getcwd(), "chatterbox_output.wav")
            wavfile.write(save_path, native_sample_rate, wav.astype('float32'))
            logger.info(f"Saved Chatterbox raw audio to {save_path}")
            
            # HIGH QUALITY: Process to required format (44.1 kHz, 16-bit, mono PCM)
            return self._process_chatterbox_audio_high_quality(wav, native_sample_rate)
            
        except Exception as e:
            logger.error(f"Chatterbox TTS generation failed: {e}")
            logger.error(f"Error type: {type(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None

    def _process_chatterbox_audio_high_quality(self, wav: np.ndarray, native_sample_rate: int) -> bytes:
        """High-quality audio processing for Chatterbox to meet required format"""
        try:
            logger.info(f"Processing Chatterbox audio: shape={wav.shape}, dtype={wav.dtype}, native_sr={native_sample_rate}")
            
            # Step 1: Preserve original dynamic range (don't over-normalize)
            max_val = np.max(np.abs(wav))
            if max_val > 1.0:
                # Only normalize if exceeding [-1, 1] range
                wav = wav / max_val
                logger.debug(f"Normalized audio to [-1, 1] range")
            
            # Step 2: High-quality resampling from 22.05kHz to 44.1kHz
            target_sample_rate = 44100
            if native_sample_rate != target_sample_rate:
                # Calculate new length for 44.1kHz
                new_length = int(len(wav) * target_sample_rate / native_sample_rate)
                
                # Use scipy for highest quality resampling
                try:
                    from scipy import signal
                    wav = signal.resample(wav, new_length)
                    logger.debug(f"Resampled from {native_sample_rate}Hz to {target_sample_rate}Hz using scipy")
                except ImportError:
                    # Fallback to librosa with high quality settings
                    wav = librosa.resample(wav, orig_sr=native_sample_rate, target_sr=target_sample_rate, 
                                         res_type='kaiser_best')  # Highest quality resampling
                    logger.debug(f"Resampled from {native_sample_rate}Hz to {target_sample_rate}Hz using librosa")
            
            # Step 3: Ensure mono (should already be mono, but double-check)
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
                logger.debug("Converted to mono")
            
            # Step 4: High-quality 16-bit PCM conversion
            # Use dithering to reduce quantization noise
            wav = self._apply_dithering(wav)
            
            # Convert to 16-bit PCM with proper scaling
            wav = (wav * 32767).astype(np.int16)
            logger.debug(f"Converted to 16-bit PCM, range: [{wav.min()}, {wav.max()}]")
            
            # Step 5: Convert to bytes
            audio_bytes = wav.tobytes()
            logger.info(f"Final audio: {len(audio_bytes)} bytes, {len(wav)} samples at {target_sample_rate}Hz")
            
            return audio_bytes
            
        except Exception as e:
            logger.error(f"High-quality Chatterbox audio processing failed: {e}")
            return None
    
    def switch_model(self, model_type: str) -> bool:
        """Switch between Kokoro and Chatterbox models dynamically"""
        try:
            if model_type not in ["kokoro", "chatterbox"]:
                logger.error(f"Unsupported model type: {model_type}")
                return False
            
            if model_type == self.model_type:
                logger.info(f"Model {model_type} is already active")
                return True
            
            logger.info(f"Switching from {self.model_type} to {model_type}...")
            
            # Update configuration file
            from config import update_config
            update_config("tts", "selected_model", model_type)
            
            # Update local configuration variable
            global tts_config
            tts_config["selected_model"] = model_type
            
            # Clear the current model to force reinitialization
            self.model = None
            self.model_type = None
            
            # Reinitialize engine with new model
            try:
                self._init_engine()
                logger.info(f"Engine reinitialization completed for {model_type}")
            except Exception as e:
                logger.error(f"Failed to reinitialize engine for {model_type}: {e}")
                # Reset to previous state
                self.model_type = None
                self.model = None
                return False
            
            # Verify the switch was successful
            if self.model_type == model_type and self.model is not None:
                logger.info(f"Successfully switched to {model_type} model")
                logger.info(f"Model object type: {type(self.model)}")
                return True
            else:
                logger.error(f"Model switch verification failed. Expected: {model_type}, Got: {self.model_type}")
                logger.error(f"Model object: {type(self.model) if self.model else 'None'}")
                return False
            
        except Exception as e:
            logger.error(f"Failed to switch to {model_type} model: {e}")
            return False
    
    def get_model_info(self) -> dict:
        """Get information about the current model"""
        return {
            "model_type": self.model_type,
            "device": self.device,
            "available_models": ["kokoro", "chatterbox"],
            "current_config": tts_config.get(self.model_type, {}) if self.model_type else {}
        }
    
    def refresh_configuration(self):
        """Refresh configuration from file and reinitialize if needed"""
        try:
            global tts_config
            from config import get_tts_config
            new_tts_config = get_tts_config()
            
            # Check if model selection changed
            if new_tts_config.get("selected_model") != self.model_type:
                logger.info(f"Configuration changed: {self.model_type} -> {new_tts_config.get('selected_model')}")
                tts_config = new_tts_config
                self._init_engine()
                return True
            else:
                logger.debug("No configuration changes detected")
                return False
                
        except Exception as e:
            logger.error(f"Failed to refresh configuration: {e}")
            return False
    
    def check_kokoro_availability(self) -> bool:
        """Check if Kokoro TTS is available and properly installed"""
        try:
            import kokoro
            logger.info("Kokoro package is importable")
            
            # Try to create a minimal KPipeline instance
            from kokoro import KPipeline
            test_pipeline = KPipeline(lang_code='a')
            logger.info("Kokoro KPipeline can be instantiated")
            
            # Check if it's callable
            if hasattr(test_pipeline, '__call__'):
                logger.info("Kokoro KPipeline is callable")
                return True
            else:
                logger.warning("Kokoro KPipeline is not callable")
                return False
                
        except ImportError as e:
            logger.error(f"Kokoro package not available: {e}")
            return False
        except Exception as e:
            logger.error(f"Kokoro initialization test failed: {e}")
            return False
    
    def generate_character_alignments(self, text: str, audio_duration_ms: float, audio_data: bytes = None) -> Dict:
        """Generate character alignment data using Forced Alignment for accurate timing"""
        if not text.strip():
            return {"chars": [], "char_start_times_ms": [], "char_durations_ms": []}
        
        try:
            # Try forced alignment first (most accurate)
            if audio_data is not None:
                logger.info("Attempting forced alignment with audio data...")
                alignments = self._forced_align_text_audio(text, audio_data)
                if alignments and len(alignments["char_start_times_ms"]) > 0:
                    logger.info("Forced alignment successful")
                    return alignments
            
            # Fallback to basic alignment if no audio data
            logger.info("Using basic alignment...")
            return self._basic_character_alignment(text, audio_duration_ms)
            
        except Exception as e:
            logger.error(f"Forced alignment failed: {e}")
            logger.warning("Falling back to basic alignment")
            return self._basic_character_alignment(text, audio_duration_ms)
    
    def _forced_align_text_audio(self, text: str, audio_data: bytes) -> Optional[Dict]:
        """Use Montreal Forced Aligner (MFA) for precise text-audio alignment"""

        import tempfile
        import os
        from montreal_forced_aligner.command_line.align import align_corpus
        
        logger.info("Initializing MFA forced alignment...")
        
        # Create temporary files for MFA
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save audio as WAV file
            audio_path = os.path.join(temp_dir, "audio.wav")
            self._save_audio_as_wav(audio_data, audio_path)
            
            # Create text file
            text_path = os.path.join(temp_dir, "text.txt")
            with open(text_path, 'w', encoding='utf-8') as f:
                f.write(text)
            
            # Create corpus directory structure for MFA
            corpus_dir = os.path.join(temp_dir, "corpus")
            os.makedirs(corpus_dir, exist_ok=True)
            
            # Move files to corpus structure
            os.rename(audio_path, os.path.join(corpus_dir, "audio.wav"))
            os.rename(text_path, os.path.join(corpus_dir, "audio.lab"))
            
            # Run MFA alignment
            logger.info("Running MFA alignment...")
            align_corpus(
                corpus_directory=corpus_dir,
                dictionary_path="english",  # Use English dictionary
                acoustic_model_path="english",  # Use English acoustic model
                output_directory=os.path.join(temp_dir, "output"),
                clean=True
            )
            
            # Parse alignment results
            alignments = self._parse_mfa_output(temp_dir, text)
            return alignments
    
    
    def _parse_mfa_output(self, temp_dir: str, text: str) -> Dict:
        """Parse MFA alignment output files"""
        try:
            import os
            from praatio import textgrid
            
            # Find TextGrid file
            output_dir = os.path.join(temp_dir, "output")
            textgrid_files = [f for f in os.listdir(output_dir) if f.endswith('.TextGrid')]
            
            if not textgrid_files:
                logger.error("No TextGrid files found in MFA output")
                return None
            
            textgrid_path = os.path.join(output_dir, textgrid_files[0])
            tg = textgrid.openTextgrid(textgrid_path)
            
            # Extract word-level alignments
            word_tier = tg.getTier("words")
            word_alignments = []
            
            for interval in word_tier.intervals:
                word_alignments.append({
                    'word': interval.label,
                    'start': interval.start * 1000,  # Convert to ms
                    'end': interval.end * 1000
                })
            
            # Convert word alignments to character alignments
            alignments = self._words_to_char_alignments(text, word_alignments)
            return alignments
            
        except Exception as e:
            logger.error(f"Failed to parse MFA output: {e}")
            return None
    

    
    def _words_to_char_alignments(self, text: str, word_alignments: list) -> Dict:
        """Convert word-level alignments to character-level"""
        chars = list(text)
        char_start_times = []
        char_durations = []
        
        char_idx = 0
        for word_info in word_alignments:
            word = word_info['word']
            word_start = word_info['start']
            word_end = word_info['end']
            word_duration = word_end - word_start
            
            # Distribute word duration across characters
            chars_in_word = len(word)
            if chars_in_word > 0:
                char_duration = word_duration / chars_in_word
                
                for i, char in enumerate(word):
                    if char_idx < len(chars) and chars[char_idx] == char:
                        char_start = word_start + (i * char_duration)
                        char_start_times.append(int(char_start))
                        char_durations.append(int(char_duration))
                        char_idx += 1
        
        return {
            "chars": chars[:len(char_start_times)],
            "char_start_times_ms": char_start_times,
            "char_durations_ms": char_durations
        }
    
    def _char_alignments_to_format(self, char_alignments: list) -> Dict:
        """Convert character alignment objects to required format"""
        chars = []
        char_start_times = []
        char_durations = []
        
        for alignment in char_alignments:
            chars.append(alignment['char'])
            char_start_times.append(int(alignment['start']))
            char_durations.append(int(alignment['end'] - alignment['start']))
        return {
            "chars": chars,
            "char_start_times_ms": char_start_times,
            "char_durations_ms": char_durations
        }
    
    def _save_audio_as_wav(self, audio_data: bytes, file_path: str):
        """Save audio bytes as WAV file for forced alignment"""
        try:
            import wave
            import numpy as np
            
            # Convert bytes back to numpy array
            # Assuming 16-bit PCM, 44.1kHz, mono
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            # Save as WAV
            with wave.open(file_path, 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(44100)  # 44.1kHz
                wav_file.writeframes(audio_array.tobytes())
                
            logger.debug(f"Saved audio to {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to save audio as WAV: {e}")
            raise
    
    def _basic_character_alignment(self, text: str, audio_duration_ms: float) -> Dict:
        """Fallback basic alignment when forced alignment fails"""
        logger.warning("Using fallback basic character alignment")
        
        if not text.strip():
            return {"chars": [], "char_start_times_ms": [], "char_durations_ms": []}
        
        chars = list(text)
        total_chars = len(chars)
        
        if total_chars == 0:
            return {"chars": [], "char_start_times_ms": [], "char_durations_ms": []}
        
        # Simple proportional distribution
        char_duration = audio_duration_ms / total_chars
        char_start_times = [int(i * char_duration) for i in range(total_chars)]
        char_durations = [int(char_duration)] * total_chars
        
        return {
            "chars": chars,
            "char_start_times_ms": char_start_times,
            "char_durations_ms": char_durations
        }

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
        logger.info(f"WebSocket connected: {connection_id} with fresh buffer")
    
    def clear_buffer(self, connection_id: str):
        """Clear the text buffer for a specific connection"""
        if connection_id in self.connection_buffers:
            self.connection_buffers[connection_id] = ""
            logger.info(f"Buffer cleared for connection: {connection_id}")
    
    def get_buffer_content(self, connection_id: str) -> str:
        """Get current buffer content for debugging"""
        return self.connection_buffers.get(connection_id, "")
    
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
                # Clear buffer before closing to prevent text accumulation
                self.clear_buffer(connection_id)
                return
            
            # Handle explicit buffer clear request
            if text == "" and flush:
                logger.info(f"Client requested buffer clear for {connection_id}")
                self.clear_buffer(connection_id)
                return
            
            # Add text to buffer (skip initial space and empty strings)
            if text and text != " ":
                # Log buffer state for debugging
                current_buffer = self.connection_buffers[connection_id]
                if current_buffer:
                    logger.debug(f"Adding text to existing buffer. Current: '{current_buffer[:30]}...', New: '{text[:30]}...'")
                else:
                    logger.debug(f"Starting new buffer with text: '{text[:30]}...'")
                
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
                
                # ALWAYS clear buffer after processing to prevent text accumulation
                self.connection_buffers[connection_id] = ""
                logger.info(f"Processed text: '{buffer_text[:50]}...' and cleared buffer for {connection_id}")
            
            # Preemptive generation for better latency (but clear buffer after each chunk)
            elif preemptive and buffer_text and len(buffer_text) > 2:
                # Start generating audio for partial text to reduce latency
                asyncio.create_task(self.generate_and_send_audio_async(websocket, connection_id, buffer_text))
                # Clear buffer after preemptive generation to prevent accumulation
                self.connection_buffers[connection_id] = ""
                logger.info(f"Preemptive generation for text: '{buffer_text[:50]}...' and cleared buffer for {connection_id}")
            
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
                
                # Calculate audio duration for 44.1 kHz, mono PCM (16-bit or 24-bit)
                # Formula: duration_ms = (bytes / (sample_rate * channels * bytes_per_sample)) * 1000
                sample_rate = 44100  # Fixed as per requirements
                channels = 1         # Fixed as per requirements (mono)
                
                # Detect bit depth from audio data size
                # 16-bit = 2 bytes per sample, 24-bit = 4 bytes per sample (aligned to 32-bit)
                total_samples = len(audio_data) // channels
                if total_samples % 4 == 0:
                    # Likely 24-bit audio (4 bytes per sample due to 32-bit alignment)
                    bytes_per_sample = 4
                    bit_depth = 24
                else:
                    # Likely 16-bit audio
                    bytes_per_sample = 2
                    bit_depth = 16
                
                sample_count = len(audio_data) // (channels * bytes_per_sample)
                duration_ms = (sample_count / sample_rate) * 1000
                
                # Generate character alignments using forced alignment with audio data
                alignments = tts_manager.generate_character_alignments(text, duration_ms, audio_data)
                
                # Send audio chunk in exact required format
                response = {
                    "audio": audio_base64,
                    "alignment": alignments
                }
                
                if self.is_connected(connection_id):
                    success = await self.safe_send_text(websocket, connection_id, json.dumps(response))
                    if success:
                        logger.info(f"Sent audio chunk for text: '{text[:50]}...' ({len(audio_data)} bytes, {bit_depth}-bit, {duration_ms:.1f}ms)")
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


# Global instances
tts_manager = TTSManager()
websocket_manager = WebSocketManager()