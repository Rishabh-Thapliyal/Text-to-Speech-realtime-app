import sys
import os
import warnings
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Suppress specific deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*LoRACompatibleLinear.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.backends.cuda.sdp_kernel.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*past_key_values.*")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import configuration
try:
    from config import get_config, get_tts_config, get_audio_config, get_server_config, get_websocket_config
    config = get_config()
    tts_config = get_tts_config()
    audio_config = get_audio_config()
    server_config = get_server_config()
    websocket_config = get_websocket_config()
except ImportError:
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

log_level = getattr(logging, server_config.get("log_level", "INFO"))
logging.basicConfig(
    level=log_level,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("app.log", mode="a", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(title="SigIQ TTS WebSocket API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    app.mount("/static", StaticFiles(directory="../frontend"), name="static")
    logger.info("Static files mounted from ../frontend directory")
except Exception as e:
    logger.warning(f"Could not mount static files: {e}")
    try:
        app.mount("/static", StaticFiles(directory="frontend"), name="static")
        logger.info("Static files mounted from frontend directory")
    except Exception as e2:
        logger.warning(f"Could not mount static files from frontend directory: {e2}")

from routes import router
from managers import tts_manager, websocket_manager, tts_config, audio_config, server_config

app.include_router(router)

if __name__ == "__main__":
    import uvicorn

    host = server_config.get("host", "0.0.0.0")
    port = server_config.get("port", 8001)
    debug = server_config.get("debug", False)

    print(f"🚀 Starting SigIQ TTS WebSocket Server")
    print(f"📍 Host: {host}")
    print(f"🔌 Port: {port}")
    print(f"🐛 Debug: {debug}")
    print(f"🎤 TTS Model: {tts_manager.model_type.upper()}")
    print(f"🎤 TTS Rate: {tts_config.get('speech_rate', 150)} WPM")
    print(f"🔊 Audio: {audio_config.get('sample_rate', 44100)}Hz, {audio_config.get('bit_depth', 16)}bit, {audio_config.get('channels', 1)}ch")
    print("=" * 50)

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level=server_config.get("log_level", "info").lower()
    )