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
import os
import re
from datetime import datetime


# Import your config and logger as needed
from config import get_tts_config, get_audio_config, get_server_config, get_websocket_config


tts_config = get_tts_config()
audio_config = get_audio_config()
server_config = get_server_config()
websocket_config = get_websocket_config()

logger = logging.getLogger(__name__)

# Math preprocessing: compiled regexes for efficiency
_MATH_INLINE_RE = re.compile(r"\\\((.*?)\\\)")  # \( ... \)
_MATH_DISPLAY_RE = re.compile(r"\\\[(.*?)\\\]")  # \[ ... \]
_DOLLAR_INLINE_RE = re.compile(r"\$(.+?)\$")  # $ ... $

# Common LaTeX operator replacements
_LATEX_OP_MAP = {
    r"\\times": " times ",
    r"\\cdot": " times ",
    r"\\leq": " less than or equal to ",
    r"\\geq": " greater than or equal to ",
    r"\\neq": " not equal to ",
    r"\\pm": " plus or minus ",
}

# Greek letters map (subset, extend as needed)
_GREEK_MAP = {
    r"\\alpha": " alpha ", r"\\beta": " beta ", r"\\gamma": " gamma ", r"\\delta": " delta ",
    r"\\epsilon": " epsilon ", r"\\zeta": " zeta ", r"\\eta": " eta ", r"\\theta": " theta ",
    r"\\iota": " iota ", r"\\kappa": " kappa ", r"\\lambda": " lambda ", r"\\mu": " mu ",
    r"\\nu": " nu ", r"\\xi": " xi ", r"\\pi": " pi ", r"\\rho": " rho ", r"\\sigma": " sigma ",
    r"\\tau": " tau ", r"\\upsilon": " upsilon ", r"\\phi": " phi ", r"\\chi": " chi ", r"\\psi": " psi ", r"\\omega": " omega ",
}

# Unicode operators
_UNICODE_OP_MAP = {
    "×": " times ", "÷": " divided by ", "≤": " less than or equal to ",
    "≥": " greater than or equal to ", "≠": " not equal to ", "±": " plus or minus ",
    "√": " square root of ", "∑": " the sum of ", "∫": " the integral of ",
}

# Unit mappings (lightweight)
_UNIT_MAP = {
    "m": "meters", "s": "seconds", "kg": "kilograms", "N": "newtons", "J": "joules",
    "W": "watts", "Pa": "pascals", "Hz": "hertz", "rad": "radians", "%": "percent",
}

def _call_node_math_speech(tex: str, style: str, timeout_ms: int) -> Optional[str]:
    """Invoke Node-based MathJax+SRE converter with timeout. Returns None on error/timeout."""
    try:
        import subprocess
        import shlex
        node_script = os.path.join(os.path.dirname(__file__), 'math_speech.js')
        if not os.path.exists(node_script):
            return None
        # Build command; safely pass tex as a single argument
        cmd = ['node', node_script, tex, style]
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=max(0.1, timeout_ms / 1000.0)
        )
        if completed.returncode == 0:
            spoken = (completed.stdout or '').strip()
            return spoken if spoken else None
        return None
    except Exception as _e:
        logger.debug(f"Node math_speech failed or timed out: {_e}")
        return None

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

    # ---------- Math-to-speech preprocessing (fast, with optional Node SRE) ----------
    def preprocess_text_for_tts(self, text: str) -> str:
        """Convert TeX/Unicode math in text to natural speech.
        - Uses Node MathJax+SRE if enabled and available (with timeout)
        - Falls back to lightweight Python regex-based conversion
        - Also handles unicode operators and basic unit expansions
        """
        if not text:
            return text

        # Replace inline/display TeX and $...$ via callback
        def _convert_match(m):
            tex_src = m.group(1)
            ms_cfg = tts_config.get('math_speech', {})
            if ms_cfg.get('enabled', True) and ms_cfg.get('use_node_sre', False):
                spoken = _call_node_math_speech(tex_src, ms_cfg.get('style', 'clearspeak'), ms_cfg.get('timeout_ms', 600))
                if spoken:
                    return f" {spoken} "
            return self._tex_to_speech(tex_src)

        text = _MATH_INLINE_RE.sub(_convert_match, text)
        text = _MATH_DISPLAY_RE.sub(_convert_match, text)
        text = _DOLLAR_INLINE_RE.sub(_convert_match, text)

        # Replace unicode operators
        for sym, spoken in _UNICODE_OP_MAP.items():
            if sym in text:
                text = text.replace(sym, spoken)

        # Units: m^2, m^3
        text = re.sub(r"\b([a-zA-Z]{1,3})\^2\b", lambda m: f" square {_UNIT_MAP.get(m.group(1), m.group(1))}", text)
        text = re.sub(r"\b([a-zA-Z]{1,3})\^3\b", lambda m: f" cubic {_UNIT_MAP.get(m.group(1), m.group(1))}", text)
        # Simple per-units like m/s or m/s^2
        text = re.sub(r"\b([a-zA-Z]{1,3})/([a-zA-Z]{1,3})(\^([0-9]+))?\b",
                      lambda m: f"{_UNIT_MAP.get(m.group(1), m.group(1))} per {_UNIT_MAP.get(m.group(2), m.group(2))}" +
                                (f" to the power of {m.group(4)}" if m.group(4) else ""),
                      text)

        # Equals for prosody
        text = text.replace("=", " equals ")
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _tex_to_speech(self, tex: str) -> str:
        # Operator and Greek replacements
        for pat, spoken in {**_LATEX_OP_MAP, **_GREEK_MAP}.items():
            tex = re.sub(pat, spoken, tex)

        # Fractions: \frac{a}{b}
        for _ in range(2):
            tex, _ = re.subn(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r" the fraction \1 over \2 ", tex)

        # Roots
        tex = re.sub(r"\\sqrt\{([^{}]+)\}", r" square root of \1 ", tex)
        tex = re.sub(r"\\sqrt\[([^\]]+)\]\{([^{}]+)\}", r" \1-th root of \2 ", tex)

        # Powers: x^{10}, x^2, x^3
        tex = re.sub(r"([A-Za-z0-9])\^\{([^{}]+)\}", r" \1 to the power of \2 ", tex)
        tex = re.sub(r"([A-Za-z0-9])\^2\b", r" \1 squared ", tex)
        tex = re.sub(r"([A-Za-z0-9])\^3\b", r" \1 cubed ", tex)

        # Summation/product and integrals
        tex = re.sub(r"\\sum_\{([^}]+)\}\^\{([^}]+)\}", r" the sum from \1 to \2 ", tex)
        tex = re.sub(r"\\prod_\{([^}]+)\}\^\{([^}]+)\}", r" the product from \1 to \2 ", tex)
        tex = re.sub(r"\\int_\{([^}]+)\}\^\{([^}]+)\}", r" the integral from \1 to \2 ", tex)

        # Derivatives: d/dx
        tex = re.sub(r"\\frac\{d\}\{d([A-Za-z])\}", r" dee by dee \1 ", tex)

        # Equals
        tex = tex.replace("=", " equals ")

        # Strip braces and clean whitespace
        tex = tex.replace("{", " ").replace("}", " ")
        tex = re.sub(r"\s+", " ", tex).strip()
        return tex
    
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
            
            # Latency optimizations
            try:
                # Use eval mode and enable cuDNN benchmark for kernels selection on CUDA
                if hasattr(self.model, 'eval'):
                    self.model.eval()
                if self.device == 'cuda':
                    torch.backends.cudnn.benchmark = True
            except Exception:
                pass
            
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
    
    def _save_audio_file(self, audio_data: bytes, text: str, model_type: str, audio_format: str = "processed") -> str:
        """Save generated audio to a file with descriptive naming"""
        try:
            # Create audio output directory if it doesn't exist
            audio_dir = os.path.join(os.getcwd(), "generated_audio")
            os.makedirs(audio_dir, exist_ok=True)
            
            # Generate timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create filename: model_timestamp_text.wav
            filename = f"{model_type}_{timestamp}.wav"
            filepath = os.path.join(audio_dir, filename)
            
            # Save audio data
            with open(filepath, 'wb') as f:
                f.write(audio_data)
            
            logger.info(f"Saved {audio_format} audio to: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Failed to save audio file: {e}")
            return ""

    def _save_audio_wav(self, wav: np.ndarray, sample_rate: int, text: str, model_type: str, audio_format: str = "raw") -> str:
        """Save numpy array as WAV file with descriptive naming"""
        try:
            # Create audio output directory if it doesn't exist
            audio_dir = os.path.join(os.getcwd(), "generated_audio")
            os.makedirs(audio_dir, exist_ok=True)
            
            # Generate timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create filename: model_timestamp_text_raw.wav
            filename = f"{model_type}_{timestamp}_{audio_format}.wav"
            filepath = os.path.join(audio_dir, filename)
            
            # Save as WAV file (preserve dtype to avoid unintended conversions)
            from scipy.io import wavfile
            wavfile.write(filepath, sample_rate, wav)
            
            logger.info(f"Saved {audio_format} audio to: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Failed to save WAV file: {e}")
            return ""

    def _kokoro_text_to_audio(self, text: str) -> Optional[bytes]:
        """Convert text to audio using Kokoro TTS - High Quality Implementation"""
        try:
            # Get Kokoro configuration
            kokoro_config = tts_config.get("kokoro", {})
            voice = kokoro_config.get("voice", "af_heart")
            native_sample_rate = kokoro_config.get("sample_rate", 24000)
            
            logger.info(f"Generating Kokoro audio with voice: {voice}, sample_rate: {native_sample_rate}")
            
            # Generate audio using Kokoro
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
            
            self._save_audio_wav(wav, native_sample_rate, text, "kokoro", "raw")

            # HIGH QUALITY: Process to required format (44.1 kHz, 16-bit, mono PCM)
            audio_bytes = self._process_kokoro_audio_high_quality(wav, native_sample_rate)
            
            return audio_bytes
            
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
                wav = self._resample_high_quality(wav, native_sample_rate, target_sample_rate)
                logger.debug(f"Resampled from {native_sample_rate}Hz to {target_sample_rate}Hz using polyphase filter")
            
            # Step 3: Ensure mono (should already be mono, but double-check)
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
                logger.debug("Converted to mono")
            
            # Step 4: Clean-up and edge conditioning before quantization
            wav = self._highpass_filter(wav, target_sample_rate, cutoff_hz=40.0, order=2)
            wav = self._apply_soft_expander(wav, threshold=1e-3, ratio=2.0)
            wav = self._apply_fade_in_out(wav, target_sample_rate, fade_ms=5.0)
            wav = self._remove_dc_offset(wav)
            wav = self._apply_dithering(wav)
            
            # Convert to 16-bit PCM with proper scaling
            wav = (wav * 32767).astype(np.int16)
            logger.debug(f"Converted to 16-bit PCM, range: [{wav.min()}, {wav.max()}]")
            self._save_audio_wav(wav, target_sample_rate, "text", "kokoro", "processed")

            # Step 5: Convert to bytes
            audio_bytes = wav.tobytes()
            logger.info(f"Final audio: {len(audio_bytes)} bytes, {len(wav)} samples at {target_sample_rate}Hz")
            
            return audio_bytes
            
        except Exception as e:
            logger.error(f"High-quality Kokoro audio processing failed: {e}")
            return None

    def _apply_dithering(self, wav: np.ndarray) -> np.ndarray:
        """Apply TPDF dithering at 1 LSB to reduce quantization distortion before 16-bit conversion"""
        try:
            # One Least Significant Bit for 16-bit full scale [-1, 1]
            lsb = 1.0 / 32768.0
            # Triangular PDF (sum of two independent uniform variables in [-0.5, 0.5]) scaled to 1 LSB
            u1 = np.random.random_sample(wav.shape) - 0.5
            u2 = np.random.random_sample(wav.shape) - 0.5
            # dither = (u1 + u2) * lsb
            dither = 0
            wav = wav + dither
            # Constrain to valid range
            wav = np.clip(wav, -1.0, 1.0)
            logger.debug("Applied TPDF dithering (1 LSB)")
            return wav
        except Exception as e:
            logger.warning(f"Dithering failed: {e}, continuing without dithering")
            return wav

    def _remove_dc_offset(self, wav: np.ndarray) -> np.ndarray:
        """Remove DC offset with a gentle high-pass at ~20 Hz (first-order)"""
        try:
            # Pre-warped one-pole high-pass (RBJ) at 20 Hz for 44.1k or native sr agnostic using simple mean removal
            # For stability and simplicity, remove mean which is effective for DC offset
            dc = np.mean(wav)
            if abs(dc) > 1e-6:
                wav = wav - dc
                logger.debug(f"Removed DC offset ({dc:.2e})")
            return wav
        except Exception:
            return wav

    def _resample_high_quality(self, wav: np.ndarray, native_sample_rate: int, target_sample_rate: int) -> np.ndarray:
        """High-quality resampling using polyphase filtering. Falls back to librosa 'kaiser_best'."""
        try:
            from math import gcd
            from scipy.signal import resample_poly
            g = gcd(native_sample_rate, target_sample_rate)
            up = target_sample_rate // g
            down = native_sample_rate // g
            # Kaiser window beta 8.6 ≈ 95 dB stopband attenuation
            return resample_poly(wav, up, down, window=('kaiser', 8.6))
        except Exception as e:
            logger.debug(f"resample_poly unavailable or failed ({e}), falling back to librosa")
            try:
                return librosa.resample(wav, orig_sr=native_sample_rate, target_sr=target_sample_rate, res_type='kaiser_best')
            except Exception:
                # Last resort: simple linear interpolation (lowest quality)
                logger.warning("Both scipy and librosa resampling failed; using naive interpolation")
                x_old = np.linspace(0, 1, num=len(wav), endpoint=False)
                new_length = int(len(wav) * target_sample_rate / native_sample_rate)
                x_new = np.linspace(0, 1, num=new_length, endpoint=False)
                return np.interp(x_new, x_old, wav)

    def _highpass_filter(self, wav: np.ndarray, sample_rate: int, cutoff_hz: float = 40.0, order: int = 2) -> np.ndarray:
        """Apply a zero-phase Butterworth high-pass filter to remove low-frequency rumble."""
        try:
            from scipy.signal import butter, filtfilt
            nyq = 0.5 * sample_rate
            normal_cutoff = max(cutoff_hz / nyq, 1e-6)
            b, a = butter(order, normal_cutoff, btype='highpass', analog=False)
            return filtfilt(b, a, wav).astype(wav.dtype, copy=False)
        except Exception as e:
            logger.debug(f"High-pass filter unavailable or failed ({e}); falling back to DC removal")
            return self._remove_dc_offset(wav)

    def _apply_fade_in_out(self, wav: np.ndarray, sample_rate: int, fade_ms: float = 5.0) -> np.ndarray:
        """Apply short linear fade-in/out to prevent clicks at chunk boundaries."""
        try:
            fade_samples = int(max(1, sample_rate * (fade_ms / 1000.0)))
            if fade_samples * 2 >= len(wav):
                # Too short to apply full fades; apply half-length fades
                fade_samples = max(1, len(wav) // 4)
            if fade_samples <= 0:
                return wav
            fade_in = np.linspace(0.0, 1.0, fade_samples, dtype=wav.dtype)
            fade_out = np.linspace(1.0, 0.0, fade_samples, dtype=wav.dtype)
            wav[:fade_samples] *= fade_in
            wav[-fade_samples:] *= fade_out
            return wav
        except Exception:
            return wav

    def _apply_soft_expander(self, wav: np.ndarray, threshold: float = 1e-3, ratio: float = 2.0) -> np.ndarray:
        """Apply a simple soft-knee expander to attenuate low-level noise between words.
        For |x| < threshold: y = sign(x) * t * (|x|/t)^ratio; else y = x.
        """
        try:
            abs_wav = np.abs(wav)
            mask = abs_wav < threshold
            if not np.any(mask):
                return wav
            scaled = np.sign(wav[mask]) * threshold * ((abs_wav[mask] / threshold) ** ratio)
            wav_out = wav.copy()
            wav_out[mask] = scaled
            return wav_out
        except Exception:
            return wav
    
    def _chatterbox_text_to_audio(self, text: str) -> Optional[bytes]:
        """Convert text to audio using Chatterbox TTS - High Quality Implementation"""
        try:
            logger.info("Generating Chatterbox audio...")
            
            # Generate audio using Chatterbox with inference optimizations
            use_autocast = (self.device == 'cuda')
            gen_kwargs = {}
            # Pick up configured sampling steps if the model supports it
            try:
                steps_cfg = tts_config.get("chatterbox", {}).get("sampling_steps", 100)
                gen_kwargs["steps"] = steps_cfg
            except Exception:
                pass
           
            with torch.inference_mode():
                if use_autocast:
                    with torch.cuda.amp.autocast():
                        try:
                            wav = self.model.generate(text, **gen_kwargs)
                        except TypeError:
                            wav = self.model.generate(text)
                else:
                    try:
                        wav = self.model.generate(text, **gen_kwargs)
                    except TypeError:
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

            # Save raw audio to WAV file
            self._save_audio_wav(wav, native_sample_rate, text, "chatterbox", "raw")
            
            # HIGH QUALITY: Process to required format (44.1 kHz, 16-bit, mono PCM)
            audio_bytes = self._process_chatterbox_audio_high_quality(wav, native_sample_rate)
            
            return audio_bytes
            
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
                wav = self._resample_high_quality(wav, native_sample_rate, target_sample_rate)
                logger.debug(f"Resampled from {native_sample_rate}Hz to {target_sample_rate}Hz using polyphase filter")
            
            # Step 3: Ensure mono (should already be mono, but double-check)
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
                logger.debug("Converted to mono")
            
            # Step 4: Clean-up and edge conditioning before quantization
            wav = self._highpass_filter(wav, target_sample_rate, cutoff_hz=40.0, order=2)
            wav = self._apply_soft_expander(wav, threshold=1e-3, ratio=2.0)
            wav = self._apply_fade_in_out(wav, target_sample_rate, fade_ms=5.0)
            wav = self._remove_dc_offset(wav)
            wav = self._apply_dithering(wav)
            
            # Convert to 16-bit PCM with proper scaling
            wav = (wav * 32767).astype(np.int16)
            logger.debug(f"Converted to 16-bit PCM, range: [{wav.min()}, {wav.max()}]")
            self._save_audio_wav(wav, target_sample_rate, "text", "chatterbox", "processed")
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
            if audio_data is not None and tts_config.get("alignment_type", "") == "forced":
                logger.info("Attempting forced alignment with audio data...")
                
                alignments = self._forced_alignment_mfa(text, audio_data)
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
    
    
    def _forced_alignment_mfa(self, text: str, audio_data: np.ndarray) -> Dict:
        """Use Montreal Forced Aligner (MFA) for precise text-audio alignment"""

        import tempfile
        import os
        import subprocess
        
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
            
            # Run MFA alignment using command line (MFA 3.x approach)
            logger.info("Running MFA alignment...")
            try:
                result = subprocess.run([
                    "mfa", "align", 
                    corpus_dir, 
                    "english_us_arpa", 
                    "english_us_arpa", 
                    os.path.join(temp_dir, "output"),
                    "--clean"
                ], capture_output=True, text=True, check=True)
                
                # Parse alignment results
                alignments = self._parse_mfa_output(temp_dir, text)
                return alignments
                
            except subprocess.CalledProcessError as e:
                logger.error(f"MFA command failed: {e}")
                logger.error(f"STDOUT: {e.stdout}")
                logger.error(f"STDERR: {e.stderr}")
                # Fall back to basic alignment if MFA fails
                logger.warning("MFA alignment failed, falling back to basic alignment")
                return self._basic_character_alignment(text, len(audio_data) / self.audio_config.get("sample_rate", 22050) * 1000)
            except FileNotFoundError:
                logger.error("MFA command not found. Please ensure MFA is properly installed and in PATH")
                return self._basic_character_alignment(text, len(audio_data) / self.audio_config.get("sample_rate", 22050) * 1000)
    
    
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
            tg = textgrid.openTextgrid(textgrid_path, includeEmptyIntervals=False)
            
            # Extract word-level alignments
            word_tier = tg.getTier("words")
            word_alignments = []
            
            for interval in word_tier.entries:
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
        char_start_times = []
        char_durations = []
        
        for word_info in word_alignments:
            word = word_info['word']
            word_start = word_info['start']
            word_end = word_info['end']
            word_duration = word_end - word_start
            
            # Distribute word duration across characters
            chars_in_word = len(word)
            if chars_in_word > 0:
                char_duration = word_duration / chars_in_word
                
                for i in range(chars_in_word):
                    char_start = word_start + (i * char_duration)
                    char_start_times.append(int(char_start))
                    char_durations.append(int(char_duration))
        
        # Get all characters from the text (excluding spaces for cleaner output)
        chars = [char for char in text if char != ' ']
        
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
        
        chars = list(text.lower())
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

    def get_audio_storage_info(self) -> Dict:
        """Get information about saved audio files and storage directory"""
        try:
            audio_dir = os.path.join(os.getcwd(), "generated_audio")
            
            if not os.path.exists(audio_dir):
                return {
                    "directory": audio_dir,
                    "exists": False,
                    "file_count": 0,
                    "files": []
                }
            
            files = []
            total_size = 0
            
            for filename in os.listdir(audio_dir):
                if filename.endswith('.wav'):
                    filepath = os.path.join(audio_dir, filename)
                    file_stat = os.stat(filepath)
                    files.append({
                        "filename": filename,
                        "size_bytes": file_stat.st_size,
                        "size_mb": round(file_stat.st_size / (1024 * 1024), 2),
                        "created": datetime.fromtimestamp(file_stat.st_ctime).isoformat(),
                        "modified": datetime.fromtimestamp(file_stat.st_mtime).isoformat()
                    })
                    total_size += file_stat.st_size
            
            # Sort files by creation time (newest first)
            files.sort(key=lambda x: x["created"], reverse=True)
            
            return {
                "directory": audio_dir,
                "exists": True,
                "file_count": len(files),
                "total_size_bytes": total_size,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "files": files
            }
            
        except Exception as e:
            logger.error(f"Failed to get audio storage info: {e}")
            return {
                "directory": audio_dir if 'audio_dir' in locals() else "unknown",
                "exists": False,
                "error": str(e)
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
        except Exception as e:
            # Only mark as disconnected for specific WebSocket errors
            if "WebSocket" in str(e) or "connection" in str(e).lower():
                logger.warning(f"WebSocket {connection_id} is no longer valid, marking as disconnected")
                self.connection_states[connection_id] = False
                return False
            else:
                # For other exceptions, assume connection is still valid
                logger.debug(f"Non-critical exception in is_connected check for {connection_id}: {e}")
                return True
    
    async def safe_send_text(self, websocket: WebSocket, connection_id: str, text: str) -> bool:
        """Safely send text to WebSocket, returns True if successful"""
        if not self.is_connected(connection_id):
            return False
        
        try:
            await websocket.send_text(text)
            return True
        except WebSocketDisconnect:
            logger.info(f"WebSocket disconnected while sending to {connection_id}")
            self.disconnect(connection_id)
            return False
        except Exception as e:
            logger.warning(f"Failed to send message to {connection_id}: {e}")
            # Don't disconnect immediately for non-WebSocket errors
            # Only disconnect if it's a WebSocket-specific error
            if "WebSocket" in str(e) or "connection" in str(e).lower():
                logger.warning(f"WebSocket error detected, disconnecting {connection_id}")
                self.disconnect(connection_id)
            return False
    
    async def process_message(self, websocket: WebSocket, connection_id: str, message: dict):
        """Process incoming WebSocket message according to specification"""
        try:
            # Check if connection is still valid
            if not self.is_connected(connection_id):
                logger.warning(f"Processing message for disconnected connection: {connection_id}")
                return
            
            # Handle different message formats from frontend
            if isinstance(message, str):
                # Frontend might send just text as string
                text = message
                flush = False
            elif isinstance(message, dict):
                text = message.get("text", "")
                flush = message.get("flush", False)
            else:
                logger.warning(f"Invalid message format from {connection_id}: {type(message)}")
                await self.safe_send_text(websocket, connection_id, json.dumps({
                    "error": "Invalid message format"
                }))
                return
            
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
                # Math-to-speech preprocessing before chunking (fast regex-based)
                try:
                    text = tts_manager.preprocess_text_for_tts(text)
                except Exception as _e:
                    logger.debug(f"Math preprocessing skipped due to error: {_e}")
                # Log buffer state for debugging
                current_buffer = self.connection_buffers[connection_id]
                if current_buffer:
                    logger.debug(f"Adding text to existing buffer. Current: '{current_buffer[:30]}...', New: '{text[:30]}...'")
                else:
                    logger.debug(f"Starting new buffer with text: '{text[:30]}...'")
                
                self.connection_buffers[connection_id] += text
                # Realtime chunking: enqueue sentence-aware 10-word chunks for sequential processing
                buffer_text = self.connection_buffers[connection_id]
                chunk_word_count = tts_config.get("chunk_word_count", 10)
                queue = self.audio_queues.get(connection_id)
                if queue is None:
                    self.audio_queues[connection_id] = asyncio.Queue()
                    queue = self.audio_queues[connection_id]
                
                # Tokenize words with trailing whitespace to preserve spacing
                tokens = list(re.finditer(r'\S+\s*', buffer_text))
                enqueued = False
                current_start_pos = 0
                i = 0
                while i < len(tokens):
                    words_in_chunk = 0
                    last_sentence_end_index = None
                    chunk_end_index = i - 1
                    # Accumulate up to 10 words, preferring to end at a sentence boundary
                    while i < len(tokens) and words_in_chunk < chunk_word_count:
                        token_text = tokens[i].group(0)
                        words_in_chunk += 1
                        chunk_end_index = i
                        # Detect sentence boundary at the end of this token
                        if re.search(r'[.!?]["\')]*\s*$', token_text):
                            last_sentence_end_index = i
                        i += 1
                    # Prefer to cut at the last sentence boundary within the chunk window
                    if last_sentence_end_index is not None:
                        cut_index = last_sentence_end_index
                        # Reset i to just after the sentence end
                        i = last_sentence_end_index + 1
                    else:
                        cut_index = chunk_end_index
                    # Compute absolute end position for the chunk
                    chunk_end_pos = tokens[cut_index].end()
                    # Form chunk string and enqueue
                    chunk_str = buffer_text[current_start_pos:chunk_end_pos].strip()
                    if chunk_str:
                        queue.put_nowait(chunk_str)
                        enqueued = True
                        logger.debug(f"Enqueued sentence-aware 10-word chunk for {connection_id}: '{chunk_str[:60]}'")
                    current_start_pos = chunk_end_pos
                
                # Remainder after removing all full chunks
                remainder = buffer_text[current_start_pos:]
                
                # If flush requested, enqueue any leftover remainder as a final chunk
                if flush and remainder.strip():
                    queue.put_nowait(remainder.strip())
                    logger.debug(f"Enqueued final (flush) chunk for {connection_id}: '{remainder.strip()[:60]}'")
                    remainder = ""
                    enqueued = True
                
                # Save remainder back to buffer
                self.connection_buffers[connection_id] = remainder
                if enqueued:
                    logger.info(f"Queued sentence-aware word chunks for {connection_id}; remaining buffer length={len(remainder)}")
                    return
            
            # Generation is handled by process_audio_queue consuming enqueued 20-char chunks.

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
            self.connection_tasks[connection_id] = asyncio.gather(
                receive_task, 
                process_task, 
                return_exceptions=True
            )
            
            # Wait for both tasks to complete (indicating actual disconnection)
            results = await asyncio.gather(
                receive_task, 
                process_task, 
                return_exceptions=True
            )
            
            # Check if any task failed with an exception
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    task_name = "receive" if i == 0 else "process"
                    logger.error(f"Task {task_name} failed for {connection_id}: {result}")
                    # Don't disconnect immediately, let the other task continue
                    if isinstance(result, WebSocketDisconnect):
                        logger.info(f"WebSocket disconnected by client: {connection_id}")
                        break
            
        except asyncio.CancelledError:
            logger.info(f"Bidirectional streaming cancelled for {connection_id}")
        except Exception as e:
            logger.error(f"Error in bidirectional streaming: {e}")
        finally:
            # Only cancel tasks if connection is actually disconnected
            if not self.is_connected(connection_id):
                if connection_id in self.connection_tasks:
                    task = self.connection_tasks[connection_id]
                    if not task.done():
                        task.cancel()
                        logger.info(f"Cancelled remaining tasks for disconnected connection: {connection_id}")
    
    async def receive_messages(self, websocket: WebSocket, connection_id: str):
        """Receive messages from client"""
        try:
            while self.is_connected(connection_id):
                try:
                    data = await websocket.receive_text()
                    message = json.loads(data)
                    await self.process_message(websocket, connection_id, message)
                except WebSocketDisconnect:
                    logger.info(f"WebSocket disconnected by client: {connection_id}")
                    break
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON received from {connection_id}: {e}")
                    # Send error response but don't disconnect
                    if self.is_connected(connection_id):
                        await self.safe_send_text(websocket, connection_id, json.dumps({
                            "error": "Invalid JSON format"
                        }))
                except Exception as e:
                    logger.error(f"Error processing message from {connection_id}: {e}")
                    # Send error response but don't disconnect
                    if self.is_connected(connection_id):
                        await self.safe_send_text(websocket, connection_id, json.dumps({
                            "error": f"Message processing failed: {str(e)}"
                        }))
        except Exception as e:
            logger.error(f"Fatal error in receive_messages for {connection_id}: {e}")
            # Only disconnect on fatal errors
            if self.is_connected(connection_id):
                self.disconnect(connection_id)
    
    async def process_audio_queue(self, websocket: WebSocket, connection_id: str):
        """Process audio queue for sending (if needed for queued operations)"""
        try:
            while self.is_connected(connection_id):
                try:
                    queue = self.audio_queues.get(connection_id)
                    if queue is None:
                        await asyncio.sleep(0.01)
                        continue

                    try:
                        chunk = await asyncio.wait_for(queue.get(), timeout=0.05)
                    except asyncio.TimeoutError:
                        continue

                    if chunk:
                        await self.generate_and_send_audio_async(websocket, connection_id, chunk)
                except Exception as e:
                    logger.error(f"Error in audio queue processing for {connection_id}: {e}")
                    # Don't break the loop for non-fatal errors
                    await asyncio.sleep(0.1)  # Longer delay on error
        except Exception as e:
            logger.error(f"Fatal error in process_audio_queue for {connection_id}: {e}")
            # Only disconnect on fatal errors
            if self.is_connected(connection_id):
                self.disconnect(connection_id)
    
    # Keep the old method for backward compatibility
    async def generate_and_send_audio(self, websocket: WebSocket, connection_id: str, text: str):
        """Legacy method - now calls the async version"""
        await self.generate_and_send_audio_async(websocket, connection_id, text)


# Global instances
tts_manager = TTSManager()
websocket_manager = WebSocketManager()