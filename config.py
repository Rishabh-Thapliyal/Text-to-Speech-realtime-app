"""
Configuration file for the SigIQ TTS WebSocket System
Modify these settings to customize the system behavior
"""

import os
from typing import Dict, Any

# TTS Engine Configuration
TTS_CONFIG = {
    # Model configuration
    "model_name": "microsoft/speecht5_tts",  # Base model, replace with your Chatterbox model path
    "model_path": None,  # Local path to Chatterbox model weights
    
    # Speech rate in words per minute (WPM)
    "speech_rate": 150,
    
    # Volume level (0.0 to 1.0)
    "volume": 0.9,
    
    # Voice ID (leave None for auto-selection)
    "voice_id": None,
    
    # Speech properties
    "properties": {
        "rate": 150,
        "volume": 0.9,
        "pitch": 1.0,
    },
    
    # Chatterbox specific settings
    "chatterbox": {
        "use_local_weights": True,  # Use local Chatterbox model weights
        "weights_path": "./chatterbox_weights",  # Path to Chatterbox weights
        "model_type": "realtime_tts",  # Model type for RealtimeTTS
        "enable_streaming": True,  # Enable real-time streaming
        "chunk_size": 50,  # Text chunk size for streaming
    }
}

# Audio Output Configuration
AUDIO_CONFIG = {
    # Target sample rate in Hz
    "sample_rate": 44100,
    
    # Bit depth
    "bit_depth": 16,
    
    # Number of channels (1 = mono, 2 = stereo)
    "channels": 1,
    
    # Audio format
    "format": "PCM",
    
    # Chunk size for processing (in characters)
    "chunk_size": 50,
    
    # Minimum text length to trigger audio generation
    "min_text_length": 10,
}

# WebSocket Configuration
WEBSOCKET_CONFIG = {
    # Maximum number of concurrent connections
    "max_connections": 100,
    
    # Connection timeout in seconds
    "connection_timeout": 30,
    
    # Message timeout in seconds
    "message_timeout": 10,
    
    # Enable CORS
    "enable_cors": True,
    
    # CORS origins (use ["*"] for development)
    "cors_origins": ["*"],
    
    # CORS methods
    "cors_methods": ["*"],
    
    # CORS headers
    "cors_headers": ["*"],
}

# Server Configuration
SERVER_CONFIG = {
    # Host to bind to (use "0.0.0.0" for all interfaces)
    "host": "0.0.0.0",
    
    # Port to listen on
    "port": 8000,
    
    # Enable debug mode
    "debug": False,
    
    # Log level
    "log_level": "INFO",
    
    # Enable auto-reload (for development)
    "auto_reload": False,
    
    # Number of worker processes
    "workers": 1,
}

# Character Alignment Configuration
ALIGNMENT_CONFIG = {
    # Enable character-level timing
    "enable_character_timing": True,
    
    # Timing algorithm: "simple", "phoneme", "ml"
    "timing_algorithm": "simple",
    
    # Minimum character duration in milliseconds
    "min_char_duration_ms": 30,
    
    # Maximum character duration in milliseconds
    "max_char_duration_ms": 200,
    
    # Default character duration in milliseconds
    "default_char_duration_ms": 70,
}

# Performance Configuration
PERFORMANCE_CONFIG = {
    # Enable audio caching
    "enable_audio_cache": True,
    
    # Maximum cache size in MB
    "max_cache_size_mb": 100,
    
    # Cache TTL in seconds
    "cache_ttl_seconds": 3600,
    
    # Enable connection pooling
    "enable_connection_pooling": True,
    
    # Pool size
    "pool_size": 10,
    
    # Enable compression
    "enable_compression": False,
}

# Logging Configuration
LOGGING_CONFIG = {
    # Log format
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    
    # Date format
    "date_format": "%Y-%m-%d %H:%M:%S",
    
    # Log file path (None for console only)
    "log_file": None,
    
    # Log file rotation
    "log_rotation": {
        "max_bytes": 10 * 1024 * 1024,  # 10MB
        "backup_count": 5,
    },
    
    # Enable structured logging
    "structured_logging": False,
}

# Development Configuration
DEV_CONFIG = {
    # Enable hot reload
    "hot_reload": False,
    
    # Enable profiling
    "enable_profiling": False,
    
    # Enable metrics collection
    "enable_metrics": False,
    
    # Development mode
    "development_mode": True,
}

def get_config() -> Dict[str, Any]:
    """Get the complete configuration dictionary"""
    return {
        "tts": TTS_CONFIG,
        "audio": AUDIO_CONFIG,
        "websocket": WEBSOCKET_CONFIG,
        "server": SERVER_CONFIG,
        "alignment": ALIGNMENT_CONFIG,
        "performance": PERFORMANCE_CONFIG,
        "logging": LOGGING_CONFIG,
        "dev": DEV_CONFIG,
    }

def get_tts_config() -> Dict[str, Any]:
    """Get TTS-specific configuration"""
    return TTS_CONFIG

def get_audio_config() -> Dict[str, Any]:
    """Get audio-specific configuration"""
    return AUDIO_CONFIG

def get_websocket_config() -> Dict[str, Any]:
    """Get WebSocket-specific configuration"""
    return WEBSOCKET_CONFIG

def get_server_config() -> Dict[str, Any]:
    """Get server-specific configuration"""
    return SERVER_CONFIG

def update_config(section: str, key: str, value: Any) -> bool:
    """Update a configuration value"""
    config_sections = {
        "tts": TTS_CONFIG,
        "audio": AUDIO_CONFIG,
        "websocket": WEBSOCKET_CONFIG,
        "server": SERVER_CONFIG,
        "alignment": ALIGNMENT_CONFIG,
        "performance": PERFORMANCE_CONFIG,
        "logging": LOGGING_CONFIG,
        "dev": DEV_CONFIG,
    }
    
    if section in config_sections:
        if key in config_sections[section]:
            config_sections[section][key] = value
            return True
        else:
            print(f"Warning: Key '{key}' not found in section '{section}'")
            return False
    else:
        print(f"Warning: Section '{section}' not found")
        return False

def load_from_env():
    """Load configuration from environment variables"""
    # TTS Configuration
    if os.getenv("TTS_SPEECH_RATE"):
        TTS_CONFIG["speech_rate"] = int(os.getenv("TTS_SPEECH_RATE"))
    
    if os.getenv("TTS_VOLUME"):
        TTS_CONFIG["volume"] = float(os.getenv("TTS_VOLUME"))
    
    if os.getenv("TTS_VOICE_ID"):
        TTS_CONFIG["voice_id"] = os.getenv("TTS_VOICE_ID")
    
    # Audio Configuration
    if os.getenv("AUDIO_SAMPLE_RATE"):
        AUDIO_CONFIG["sample_rate"] = int(os.getenv("AUDIO_SAMPLE_RATE"))
    
    if os.getenv("AUDIO_CHUNK_SIZE"):
        AUDIO_CONFIG["chunk_size"] = int(os.getenv("AUDIO_CHUNK_SIZE"))
    
    # Server Configuration
    if os.getenv("SERVER_HOST"):
        SERVER_CONFIG["host"] = os.getenv("SERVER_HOST")
    
    if os.getenv("SERVER_PORT"):
        SERVER_CONFIG["port"] = int(os.getenv("SERVER_PORT"))
    
    if os.getenv("SERVER_DEBUG"):
        SERVER_CONFIG["debug"] = os.getenv("SERVER_DEBUG").lower() == "true"
    
    # Logging Configuration
    if os.getenv("LOG_LEVEL"):
        SERVER_CONFIG["log_level"] = os.getenv("LOG_LEVEL").upper()
    
    if os.getenv("LOG_FILE"):
        LOGGING_CONFIG["log_file"] = os.getenv("LOG_FILE")

def print_config():
    """Print the current configuration"""
    print("🔧 SigIQ TTS System Configuration")
    print("=" * 50)
    
    config = get_config()
    for section, settings in config.items():
        print(f"\n📋 {section.upper()}:")
        for key, value in settings.items():
            if isinstance(value, dict):
                print(f"   {key}:")
                for sub_key, sub_value in value.items():
                    print(f"     {sub_key}: {sub_value}")
            else:
                print(f"   {key}: {value}")

# Load configuration from environment variables on import
load_from_env()

if __name__ == "__main__":
    print_config()
