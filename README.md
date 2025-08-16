# SigIQ TTS WebSocket System

A bidirectional streaming WebSocket Text-to-Speech (TTS) system with real-time audio generation and character alignment, similar to the ElevenLabs WebSocket API.

## 📋 WebSocket Input/Output Specification

### **Input Format**
The client streams JSON chunks to the server via WebSocket with these fields:

- **`text`**: A string for which audio will be generated
  - **First chunk**: Contains only a single space character `" "`
  - **Final chunk**: Empty string `""` to close the WebSocket
  - **Other chunks**: Actual text content
- **`flush`**: Boolean that forces audio generation for buffered text
  - **`true`**: WebSocket remains open regardless of text value
  - **`false`**: Normal processing

### **Output Format**
The server streams audio chunks back via the same WebSocket:

- **`audio`**: Base64 encoding of 44.1 kHz, 16-bit, mono PCM audio
- **`alignment`**: Character alignment data with timestamps

### **Character Alignment Format**
```json
{
  "chars": ["T", "h", "i", "s", " ", "i", "s", " ", "a", "n", " ", "e", "x", "a", "m", "p", "l", "e", ".", " "],
  "char_start_times_ms": [0, 70, 139, 186, 221, 279, 325, 360, 406, 441, 476, 534, 580, 662, 755, 824, 894, 952, 1010],
  "char_durations_ms": [70, 69, 46, 35, 58, 45, 34, 46, 34, 34, 58, 45, 82, 92, 68, 70, 57, 58, 46]
}
```

### **Audio Format**
- **Sample Rate**: 44.1 kHz (fixed)
- **Bit Depth**: 16-bit (fixed)
- **Channels**: Mono (1 channel, fixed)
- **Encoding**: PCM
- **Transmission**: Base64 encoded via WebSocket

## 🚀 Features

- **Bidirectional WebSocket Streaming**: Real-time text input and audio output
- **Low Latency**: Minimized delay between input and audio generation
- **Character Alignment**: Precise timing data for each character spoken
- **Audio Format**: 44.1 kHz, 16-bit, mono PCM audio encoded in Base64
- **Modern Web Interface**: Beautiful, responsive testing client
- **Real-time Captions**: Live character highlighting synchronized with audio
- **Concurrent Processing**: Multiple WebSocket connections supported
- **Chatterbox TTS Integration**: High-quality text-to-speech using the Chatterbox model

## 🏗️ Architecture

```
┌─────────────────┐    WebSocket    ┌─────────────────┐    TTS Engine    ┌─────────────────┐
│   Frontend      │ ◄──────────────► │   Backend       │ ◄──────────────► │   Chatterbox    │
│   Client        │                 │   FastAPI       │                 │   TTS           │
│                 │                 │   Server        │                 │                 │
└─────────────────┘                 └─────────────────┘                 └─────────────────┘
```

## 📋 Requirements

- Python 3.11+
- Modern web browser with WebSocket support
- Audio playback capabilities

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd sigiq-tts
```

### 2. Set Up Python Environment
```bash
# Create and activate conda environment
conda create -n tts python=3.11.13
conda activate tts

# Or use virtualenv
python -m venv tts-env
source tts-env/bin/activate  # On Windows: tts-env\Scripts\activate
```

### 3. Install Backend Dependencies
```bash
cd backend
pip install -r requirements.txt
```

### 4. Install System Dependencies (macOS)
```bash
# Install espeak for TTS engine (optional, for fallback)
brew install espeak

# Or install via conda
conda install -c conda-forge espeak
```

### 5. Test Chatterbox TTS Integration
```bash
# Test the Chatterbox TTS integration
python test_chatterbox.py

# This will generate a test audio file to verify everything is working
```

## 🚀 Running the System

### 1. Start the Backend Server
```bash
cd backend
python main.py
```

The server will start on `http://localhost:8000`

### 2. Open the Frontend Client
Open `frontend/index.html` in your web browser, or serve it using a local server:

```bash
cd frontend
python -m http.server 8080
# Then open http://localhost:8080
```

### 3. Connect and Test
1. Click "Connect" to establish WebSocket connection
2. Enter text in the input area
3. Click "Stream Text" to start TTS generation
4. Watch real-time captions and listen to generated audio

## 🔧 Configuration

### WebSocket Endpoint
- **URL**: `ws://localhost:8000/ws/tts`
- **Protocol**: WebSocket over HTTP/HTTPS

### TTS Engine Settings
- **Speech Rate**: 150 WPM (configurable in `config.py`)
- **Volume**: 90% (configurable)
- **Voice**: Auto-selected from available system voices

### Chatterbox Model Integration
- **Model Type**: RealtimeTTS with Chatterbox weights
- **Local Weights**: Support for custom model weights
- **Streaming**: Real-time audio generation
- **Fallback**: Automatic fallback to basic TTS if model unavailable

### Audio Format
- **Sample Rate**: 44.1 kHz
- **Bit Depth**: 16-bit
- **Channels**: Mono
- **Encoding**: PCM → Base64

## 📡 API Reference

### WebSocket Messages

#### Client → Server
```json
{
  "text": "string",     // Text to convert (empty string to close)
  "flush": boolean      // Force audio generation for buffered text
}
```

#### Server → Client
```json
{
  "audio": "base64_string",           // Base64-encoded PCM audio
  "alignment": {
    "chars": ["T", "h", "i", "s"],    // Character array
    "char_start_times_ms": [0, 70],   // Start time for each character
    "char_durations_ms": [70, 69]     // Duration for each character
  }
}
```

### HTTP Endpoints

- `GET /` - API information
- `GET /health` - Health check
- `GET /docs` - Interactive API documentation (Swagger UI)

## 🎯 Usage Examples

### Basic Text Streaming
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/tts');

// Send initial space character
ws.send(JSON.stringify({text: " ", flush: false}));

// Stream text in chunks
ws.send(JSON.stringify({text: "Hello world", flush: false}));

// Force audio generation
ws.send(JSON.stringify({text: "", flush: true}));

// Close connection
ws.send(JSON.stringify({text: "", flush: false}));
```

### Chatterbox Model Integration
```python
# In config.py, configure Chatterbox settings
TTS_CONFIG = {
    "chatterbox": {
        "use_local_weights": True,
        "weights_path": "./chatterbox_weights",
        "model_type": "realtime_tts",
        "enable_streaming": True,
        "chunk_size": 50
    }
}

# The system will automatically:
# 1. Load your local Chatterbox model weights
# 2. Use RealtimeTTS for high-quality audio generation
# 3. Fall back to basic TTS if the model is unavailable
```

### Real-time Caption Display
```javascript
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  
  if (data.alignment) {
    // Display characters with timing
    data.alignment.chars.forEach((char, index) => {
      const startTime = data.alignment.char_start_times_ms[index];
      const duration = data.alignment.char_durations_ms[index];
      
      // Create timed caption display
      setTimeout(() => highlightCharacter(char), startTime);
    });
  }
};
```

## 🔍 Testing

### Manual Testing
1. **Connection Test**: Verify WebSocket connection establishment
2. **Text Streaming**: Test various text lengths and content
3. **Audio Quality**: Check audio clarity and timing
4. **Caption Sync**: Verify character highlighting matches audio
5. **Error Handling**: Test with invalid inputs and network issues

### Performance Testing
- **Latency**: Measure time from text input to audio output
- **Throughput**: Test with high-frequency text streaming
- **Concurrency**: Multiple simultaneous connections
- **Memory Usage**: Monitor resource consumption

## 🐛 Troubleshooting

### Common Issues

#### TTS Engine Not Working
```bash
# Check espeak installation
espeak --version

# Verify Python TTS library
python -c "import pyttsx3; print('TTS available')"
```

#### WebSocket Connection Failed
- Check if backend server is running
- Verify WebSocket URL format
- Check firewall/network settings
- Ensure CORS is properly configured

#### Audio Not Playing
- Check browser audio permissions
- Verify Web Audio API support
- Check console for JavaScript errors
- Ensure audio context is initialized

#### High Latency
- Reduce text chunk size
- Optimize TTS engine settings
- Check system performance
- Monitor network latency

### Debug Mode
Enable detailed logging by modifying the logging level in `backend/main.py`:
```python
logging.basicConfig(level=logging.DEBUG)
```

## 🔮 Future Enhancements

- **Advanced TTS Models**: Integration with neural TTS engines
- **Voice Selection**: Multiple voice options and customization
- **Audio Effects**: Pitch, speed, and tone adjustments
- **Streaming Optimization**: Adaptive chunk sizing and buffering
- **Authentication**: Secure WebSocket connections
- **Metrics Dashboard**: Real-time performance monitoring
- **Mobile Support**: Responsive design and touch controls

## 📚 Technical Details

### Character Alignment Algorithm
The current implementation uses a simple time-distribution algorithm:
1. Calculate total audio duration from PCM data
2. Distribute time evenly among characters
3. Generate start times and durations for each character

For production use, consider implementing:
- Phoneme-based alignment
- Machine learning models for timing prediction
- Integration with speech recognition systems

### Audio Processing Pipeline
1. **Text Input** → TTS Engine (pyttsx3)
2. **WAV Generation** → Temporary file creation
3. **Format Conversion** → 44.1kHz, 16-bit, mono PCM
4. **Base64 Encoding** → WebSocket transmission
5. **Client Decoding** → Web Audio API processing
6. **Real-time Playback** → Synchronized with captions

### WebSocket Management
- **Connection Pooling**: Multiple concurrent connections
- **Message Buffering**: Efficient text accumulation
- **Error Handling**: Graceful failure recovery
- **Resource Cleanup**: Automatic connection management

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests and documentation
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **pyttsx3**: Cross-platform TTS library
- **FastAPI**: Modern Python web framework
- **Web Audio API**: Browser audio processing
- **espeak**: Open-source speech synthesizer

## 📞 Support

For questions, issues, or contributions:
- Create an issue on GitHub
- Contact the development team
- Check the documentation and examples

---

**Happy Text-to-Speech Streaming! 🎤✨**
