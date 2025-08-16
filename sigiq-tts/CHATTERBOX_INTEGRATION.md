# Chatterbox Model Integration Guide

This guide explains how to integrate your custom Chatterbox open source model weights with the SigIQ TTS WebSocket System using RealtimeTTS.

## 🎯 Overview

The system has been updated to support:
- **RealtimeTTS**: High-quality, real-time text-to-speech generation
- **Chatterbox Models**: Integration with custom open source TTS model weights
- **Local Weights**: Support for locally stored model files
- **Automatic Fallback**: Graceful degradation if models are unavailable

## 🏗️ Architecture Changes

### Before (pyttsx3)
```
Text Input → pyttsx3 → Basic TTS → Audio Output
```

### After (RealtimeTTS + Chatterbox)
```
Text Input → RealtimeTTS → Chatterbox Model → High-Quality Audio Output
                    ↓
            Fallback TTS (if needed)
```

## 📦 New Dependencies

The system now requires additional packages for Chatterbox integration:

```bash
# Core ML libraries
torch>=2.0.0
torchaudio>=2.0.0
transformers>=4.30.0

# Audio processing
soundfile>=0.12.1
librosa>=0.10.0

# Remove old dependency
# pyttsx3==2.90  # No longer needed
```

## 🚀 Quick Start

### 1. Install New Dependencies
```bash
cd backend
pip install -r requirements.txt
```

### 2. Setup Chatterbox Integration
```bash
# Run the setup script
python setup_chatterbox.py

# This will:
# - Create necessary directories
# - Update configuration files
# - Check environment compatibility
```

### 3. Download a Model (Optional)
```bash
# Download a pre-trained model from Hugging Face
python setup_chatterbox.py --download-model "microsoft/speecht5_tts"

# Or use your own model weights (see Custom Models section)
```

### 4. Start the System
```bash
./start.sh
```

## ⚙️ Configuration

### Chatterbox Settings in `config.py`

```python
TTS_CONFIG = {
    # Model configuration
    "model_name": "microsoft/speecht5_tts",  # Default model
    "model_path": None,  # Local model path
    
    # Chatterbox specific settings
    "chatterbox": {
        "use_local_weights": True,           # Use local weights
        "weights_path": "./chatterbox_weights",  # Path to weights
        "model_type": "realtime_tts",        # Model type
        "enable_streaming": True,            # Enable streaming
        "chunk_size": 50,                   # Text chunk size
    }
}
```

### Environment Variables

You can also configure via environment variables:

```bash
export TTS_MODEL_NAME="your/model/name"
export TTS_USE_LOCAL_WEIGHTS="true"
export TTS_WEIGHTS_PATH="./your/weights/path"
```

## 📁 Model Directory Structure

Your Chatterbox model weights should be organized as follows:

```
./chatterbox_weights/
├── config.json              # Model configuration
├── pytorch_model.bin        # Model weights (or multiple .bin files)
├── tokenizer.json           # Tokenizer configuration
├── tokenizer_config.json    # Tokenizer settings
├── model_info.json          # Model metadata (optional)
└── README.md                # Model documentation (optional)
```

### Required Files

- **`config.json`**: Model architecture and parameters
- **`pytorch_model.bin`**: Trained model weights
- **`tokenizer.json`**: Text tokenization configuration

### Optional Files

- **`model_info.json`**: Metadata about the model
- **`README.md`**: Usage instructions and model information

## 🔧 Custom Model Integration

### 1. Prepare Your Model

Ensure your model is compatible with the Transformers library:

```python
from transformers import AutoModelForTextToSpeech, AutoTokenizer

# Your model should inherit from a TTS model class
class YourChatterboxModel(AutoModelForTextToSpeech):
    def __init__(self, config):
        super().__init__(config)
        # Your model implementation
    
    def generate_speech(self, input_ids, speaker_embeddings):
        # Your speech generation logic
        pass
```

### 2. Export Model Weights

```python
# Save your model and tokenizer
model.save_pretrained("./chatterbox_weights")
tokenizer.save_pretrained("./chatterbox_weights")

# Create model info
model_info = {
    "model_name": "your_chatterbox_model",
    "model_type": "text_to_speech",
    "framework": "transformers",
    "description": "Your custom Chatterbox TTS model"
}

with open("./chatterbox_weights/model_info.json", 'w') as f:
    json.dump(model_info, f, indent=2)
```

### 3. Update Configuration

```python
# In config.py
TTS_CONFIG["chatterbox"]["weights_path"] = "./chatterbox_weights"
TTS_CONFIG["chatterbox"]["use_local_weights"] = True
```

## 🧪 Testing Integration

### 1. Check Model Loading

Start the system and look for this log message:
```
✅ RealtimeTTS engine initialized successfully with Chatterbox model
```

### 2. Test TTS Generation

```bash
# Run the demo script
python demo.py

# Or test directly
python test_tts.py
```

### 3. Verify Audio Quality

- Check that audio is generated successfully
- Verify character alignment data is correct
- Test streaming functionality

## 🔍 Troubleshooting

### Common Issues

#### Model Not Loading
```
❌ Failed to initialize TTS engine: [Error details]
```

**Solutions:**
- Check model file structure
- Verify model compatibility with Transformers
- Check file permissions
- Ensure all required files are present

#### Import Errors
```
❌ Transformers not available. Install with: pip install transformers
```

**Solutions:**
```bash
pip install transformers torch torchaudio
```

#### Memory Issues
```
❌ CUDA out of memory
```

**Solutions:**
- Use CPU instead of GPU: Set `device = "cpu"`
- Reduce model size or use quantization
- Increase system memory

#### Audio Generation Fails
```
❌ RealtimeTTS generation failed: [Error details]
```

**Solutions:**
- Check model input/output format
- Verify tokenizer compatibility
- Test with simpler text inputs

### Debug Mode

Enable detailed logging:

```python
# In config.py
SERVER_CONFIG["log_level"] = "DEBUG"
```

Or via environment:
```bash
export LOG_LEVEL=DEBUG
```

## 📊 Performance Optimization

### GPU Acceleration

```python
# The system automatically detects and uses GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

### Model Caching

```python
# Enable model caching in config.py
PERFORMANCE_CONFIG["enable_audio_cache"] = True
PERFORMANCE_CONFIG["max_cache_size_mb"] = 100
```

### Streaming Optimization

```python
# Adjust chunk size for optimal performance
TTS_CONFIG["chatterbox"]["chunk_size"] = 100  # Larger chunks for better throughput
```

## 🔮 Advanced Features

### Multi-Model Support

```python
# Support multiple Chatterbox models
TTS_CONFIG["chatterbox"]["models"] = {
    "default": "./chatterbox_weights",
    "high_quality": "./high_quality_weights",
    "fast": "./fast_weights"
}
```

### Model Switching

```python
# Switch models at runtime
tts_manager.switch_model("high_quality")
```

### Custom Preprocessing

```python
# Add custom text preprocessing
def preprocess_text(text):
    # Your preprocessing logic
    return processed_text

# Integrate with TTS pipeline
tts_manager.set_preprocessor(preprocess_text)
```

## 📚 Examples

### Basic Integration

See `examples/chatterbox_integration.py` for a complete working example.

### Configuration Examples

```python
# Minimal configuration
TTS_CONFIG["chatterbox"]["use_local_weights"] = True
TTS_CONFIG["chatterbox"]["weights_path"] = "./my_model"

# Advanced configuration
TTS_CONFIG["chatterbox"] = {
    "use_local_weights": True,
    "weights_path": "./chatterbox_weights",
    "model_type": "realtime_tts",
    "enable_streaming": True,
    "chunk_size": 50,
    "enable_cache": True,
    "max_cache_size": 100,
    "device": "auto"  # auto, cpu, cuda
}
```

## 🎉 Success Indicators

Your Chatterbox integration is working correctly when you see:

1. ✅ **Model Loading**: `RealtimeTTS engine initialized successfully with Chatterbox model`
2. ✅ **Audio Generation**: High-quality audio output with proper timing
3. ✅ **Character Alignment**: Accurate character-level timing data
4. ✅ **Streaming**: Real-time text-to-speech generation
5. ✅ **Fallback**: Graceful degradation if model fails

## 📞 Support

For issues with Chatterbox integration:

1. Check the troubleshooting section above
2. Verify your model compatibility
3. Check system logs for detailed error messages
4. Test with the provided examples
5. Ensure all dependencies are properly installed

---

**🎤 Happy Chatterbox TTS Integration! 🎤**

Your custom open source model weights are now powering high-quality, real-time text-to-speech generation!
