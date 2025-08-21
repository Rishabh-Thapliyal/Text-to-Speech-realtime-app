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
  "chars": ["M", "y", " ", "n", "a", "m", "e", " ", "i", "s", " ", "R", "i", "s", "h", "a", "b", "h"],
  "char_start_times_ms": [0, 45, 89, 134, 178, 223, 267, 312, 356, 401, 445, 490, 534, 578, 623, 667, 712, 756],
  "char_durations_ms": [45, 44, 45, 44, 45, 44, 45, 44, 45, 44, 45, 44, 44, 45, 44, 44, 44, 44]
}
```

## 🎵 **Audio Quality & Processing**

### **Raw vs Processed Output Quality**

The system generates audio in two stages, each with different quality characteristics:

#### **1. Raw Model Output**
- **High Quality**: Direct output from the TTS model (Chatterbox/Kokoro)
- **Native Format**: Model's optimal audio format and quality
- **Full Fidelity**: Preserves all model-specific audio characteristics
- **No Processing Loss**: Maintains original model output quality

#### **2. Processed Output (Forced Alignment)**
- **Lower Quality**: Audio processed through forced alignment pipeline
- **Format Conversion**: Converted to 44.1 kHz, 16-bit, mono PCM
- **Processing Loss**: Quality reduction due to format standardization
- **Alignment Benefits**: Precise character timing at the cost of audio quality

#### **Quality Trade-offs**
```
Raw Model Output          Processed Output
├── 🎯 High Quality      ├── 🎯 Precise Timing
├── 🎵 Native Format     ├── 🎵 Standardized Format
├── 🔒 Full Fidelity     ├── 🔒 Reduced Quality
└── ⚠️ No Alignment      └── ✅ Character Alignment
```

#### **Why Processed Output Has Lower Quality**
1. **Sample Rate Conversion**: Models may use different sample rates (e.g., 24kHz) converted to 44.1kHz
2. **Bit Depth Reduction**: Models may use 24-bit or 32-bit internally, converted to 16-bit
3. **Format Standardization**: Conversion to PCM format may introduce quantization artifacts
4. **Forced Alignment Processing**: MFA processing may modify audio characteristics
5. **WebSocket Transmission**: Base64 encoding and transmission overhead

#### **Recommendations**
- **For Audio Quality**: Use raw model output when character alignment isn't critical
- **For Timing Accuracy**: Use processed output when precise character synchronization is needed
- **Hybrid Approach**: Consider using both outputs for different use cases

### **Audio Format**
- **Sample Rate**: 44.1 kHz (fixed)
- **Bit Depth**: 16-bit (fixed)
- **Channels**: Mono (1 channel, fixed)
- **Encoding**: PCM
- **Transmission**: Base64 encoded via WebSocket

### ✅ High-Fidelity Processing Pipeline (Update)
The processed output is now engineered to sound virtually identical to the raw model output while still meeting the required 44.1 kHz, 16‑bit PCM format:
- **Polyphase Resampling**: High-quality Kaiser-windowed polyphase resampling (with fallback to `librosa` kaiser_best).
- **DC Offset Removal**: Removes bias to prevent headroom loss and low‑freq artifacts.
- **High‑Pass Filter**: Gentle zero‑phase Butterworth HPF (~40 Hz) to eliminate rumble.
- **Short Edge Fades**: 5 ms fade‑in/out on each chunk to prevent boundary clicks.
- **Soft Expander**: Light noise gating between words to reduce inter‑word hiss without harming tails.
- **Proper Dithering**: 1 LSB TPDF dither applied just before 16‑bit quantization.
- **WAV Saving Without Forced Float**: Preserves dtype when writing intermediate WAVs to avoid needless conversions.
- **Frontend PCM Fix**: Correct little‑endian signed 16‑bit decode in the Web Audio path.

Result: processed audio is now near‑indistinguishable from the original model output in typical listening, while keeping strict format compliance and alignment support.

## 🧮 Math Expressions → Speech

### **What is supported**
- **Inline TeX**: `\( ... \)`
- **Display TeX**: `\[ ... \]`
- **Dollar inline**: `$ ... $`

Only the content inside these markers is treated as math. Everything else is spoken as normal text.

### **How it works (pipeline)**
1. **Detection**: The backend scans text for math segments (`backend/managers.py`).
2. **Primary path (optional, high‑fidelity)**: If enabled, each TeX segment is sent to a small Node helper (`backend/math_speech.js`) that uses MathJax 3 + Speech Rule Engine (SRE) to convert TeX → MathML → natural language speech.
   - Styles: `clearspeak` (default) or `mathspeak`.
   - Timeout‑guarded. If it fails or times out, we fall back transparently.
3. **Fallback path (fast, dependency‑free)**: A Python converter speaks common TeX constructs via regex rules:
   - Fractions: `\frac{a}{b}` → “the fraction a over b”
   - Roots: `\sqrt{x}` / `\sqrt[n]{x}` → “square root of x” / “n‑th root of x”
   - Exponents: `x^{10}`, `x^2`, `x^3` → “x to the power of 10”, “x squared”, “x cubed”
   - Sums/products/integrals with limits
   - Derivatives like `\frac{d}{dx}`
   - Greek letters and common operators (e.g., `\alpha`, `\beta`, `\times`, `\cdot`)
   - Braces removed; `=` normalized to “equals”
4. **Unicode operators & units**: Outside of TeX, we normalize symbols and simple units for clarity:
   - Operators: ×, ·, −, ±, ≤, ≥, ∑, ∫, ∞, etc. → spoken equivalents
   - Units: `m^2`/`m^3` → “square meters”/“cubic meters”; `m/s^2` → “meters per second to the power of 2”

### **Configuration**
Set in `config.py` under `TTS_CONFIG["math_speech"]`:
```python
"math_speech": {
    "enabled": True,          # Turn math→speech preprocessing on/off
    "use_node_sre": False,    # Use Node MathJax+SRE for high-fidelity speech
    "style": "clearspeak",   # "clearspeak" | "mathspeak"
    "timeout_ms": 6000        # Per-segment timeout for the Node helper
}
```

### **Enable the Node MathJax+SRE path (optional)**
This provides the most accurate math speech, especially for complex TeX.
```bash
# 1) Ensure Node.js is installed (v16+ recommended)
# 2) Install helper dependencies
cd backend
npm install

# (Optional) Quick test
node math_speech.js "\\frac{a}{b}" clearspeak
```
Then set `use_node_sre: True` in `config.py`.

Dependencies used by the helper (`backend/package.json`): `mathjax-full`, `speech-rule-engine`.

### **Examples**
- Input: `The equation \\(E = mc^2\\) is famous.`
  - Spoken: “The equation E equals m c squared is famous.”
- Input: `Compute \\( \\frac{a+b}{\\sqrt{c}} \\) quickly.`
  - Spoken: “Compute the fraction a plus b over square root of c quickly.”
- Input: `Gravity ≈ 9.8 m/s^2.`
  - Spoken: “Gravity approximately equals 9 point 8 meters per second to the power of 2.”

Notes:
- If the Node helper is disabled/unavailable or times out, the Python fallback is used automatically.
- Very complex TeX may be simplified in the fallback path; enable the Node path for best fidelity.

## 🎭 **Model Switching**

The system now supports **two TTS models** with seamless switching:

### **Available Models**
- **Chatterbox TTS**: High-quality, real-time TTS with streaming support
- **Kokoro TTS**: Lightweight, fast TTS with 82M parameters and Apache license

### **Configuration**
Models can be configured in `config.py`:
```python
TTS_CONFIG = {
    "selected_model": "chatterbox",  # or "kokoro"
    "kokoro": {
        "lang_code": "a",
        "voice": "af_heart",
        "sample_rate": 24000
    },
    "chatterbox": {
        "enable_streaming": True,
        "chunk_size": 50
    }
}
```

### **Dynamic Switching**
Models can be switched at runtime without restarting the server:
```python
# Switch models programmatically
import requests
response = requests.post("http://localhost:8001/models/switch/kokoro")
```

## 🎨 **Model Selection UI**

### **Interactive Model Selection**
The frontend now includes a user-friendly model selection interface:

- **🎭 Model Dropdown**: Choose between Chatterbox and Kokoro models
- **🔄 Switch Model Button**: Instantly switch models when connected
- **🔍 Refresh Status Button**: Check current model status
- **📊 Visual Status Indicator**: Real-time model status display

### **UI Features**
```
🎭 TTS Model Selection
├── 🔵 Chatterbox TTS (High Quality)
├── 🟢 Kokoro TTS (Fast & Lightweight)
├── 🔄 Switch Model (enabled when connected)
└── 🔍 Refresh Status
```

### **How to Use the UI**
1. **Connect** to the WebSocket server
2. **Select** your preferred model from the dropdown
3. **Click** "🔄 Switch Model" to switch
4. **Watch** the status indicator change in real-time
5. **Use** "🔍 Refresh Status" to verify current model

### **Status Indicators**
- **🔵 Chatterbox Active**: Blue indicator for Chatterbox model
- **🟢 Kokoro Active**: Green indicator for Kokoro model
- **❓ Unknown**: Gray indicator when status is unclear


## 🧹 **Buffer Management**

### **Buffer Clearing**
The system clears text buffers between requests to prevent text accumulation:

- **Automatic Clearing**: Buffers are cleared after each TTS generation
- **Manual Clearing**: Use the "🧹 Clear Buffer" button in the UI
- **API Endpoints**: Programmatic buffer management available


### **Text Processing Behavior**
- **Each Request**: Processes only the new text, not accumulated text
- **No Duplication**: Previous text chunks are not included in new requests
- **Clean State**: Fresh buffer for each new TTS request
- **Proper Streaming**: Text chunks are processed sequentially with delays

## 🔍 **Forced Alignment & Character Timing**

### **MFA (Montreal Forced Aligner) Integration**
The system now uses **Montreal Forced Aligner (MFA)** for precise text-audio alignment:

- **High Accuracy**: Phoneme-level alignment using MFA 3.x
- **Professional Grade**: Industry-standard forced alignment tool
- **Fallback Support**: Generic alignment when MFA is unavailable
- **Real-time Processing**: Integrated into the streaming pipeline

### **Alignment Methods**

#### **1. MFA Forced Alignment (Primary)**
#### **2. Generic Alignment (Fallback)**

### **Alignment Quality Comparison**
```
MFA Forced Alignment          Generic Alignment
├── 🎯 Phoneme-level         ├── 🎯 Character-level
├── 📊 Professional Grade     ├── 📊 Basic Proportional
├── ⏱️ Precise Timing        ├── ⏱️ Approximate Timing
├── 🔧 MFA Dependency        ├── 🔧 No Dependencies
└── ⚡ Slower Processing     └── ⚡ Fast Processing
```

### **Installation & Setup**
```bash
# Install MFA for forced alignment
cd backend

# Or manually install MFA
sh setup.sh
```

## 🎵 **Output Section & Audio Display**

### **Generated Output Display**
The system now shows both raw and processed audio outputs:

#### **Raw Model Output**
- **File Naming**: `{model}_{timestamp}_raw.wav`
- **Quality**: High-fidelity, native model format
- **Use Case**: When audio quality is priority over timing accuracy

#### **Processed Output**
- **File Naming**: `{model}_{timestamp}_processed.wav`
- **Quality**: Standardized format (44.1kHz, 16-bit, mono PCM)
- **Use Case**: When character alignment and timing accuracy are priority

### **Output File Structure**
```
generated_audio/
├── chatterbox_20250820_175448_raw.wav      # Raw model output
├── chatterbox_20250820_175448_processed.wav # Processed for alignment
├── kokoro_20250820_175449_raw.wav          # Raw Kokoro output
└── kokoro_20250820_175449_processed.wav    # Processed Kokoro output
```

### **Audio Quality Monitoring**
The system provides real-time feedback on audio generation:

- **Chunk Count**: Number of audio chunks generated
- **Total Audio**: Cumulative audio data size
- **Duration**: Current audio playback duration
- **Status**: Real-time generation status

## 🏗️ Architecture

```
┌─────────────────┐    WebSocket    ┌─────────────────┐    TTS Engine    ┌─────────────────┐
│   Frontend      │ ◄──────────────► │   Backend       │ ◄──────────────► │   Chatterbox    │
│   Client        │                 │   FastAPI       │                 │   TTS           │
│                 │                 │   Server        │                 │                 │
└─────────────────┘                 └─────────────────┘                 └─────────────────┘
                                        │
                                        ▼
                                ┌─────────────────┐
                                │   MFA Manager   │
                                │   Forced        │
                                │   Alignment     │
                                └─────────────────┘
                                        │
                                        ▼
                                ┌─────────────────┐
                                │   Audio        │
                                │   Processing   │
                                │   Pipeline     │
                                └─────────────────┘
```

## 📋 Requirements

- Python 3.11+
- Modern web browser with WebSocket support
- Audio playback capabilities
- Montreal Forced Aligner (optional, for enhanced alignment)

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
sh setup.sh
```

**Note**: The system supports two TTS engines:
- **Chatterbox TTS**: Installed via `chatterbox-tts` package
- **Kokoro TTS**: Installed via `kokoro>=0.9.2` package

Both engines will be installed automatically with the requirements.txt file.

### 4. Install System Dependencies (macOS)
```bash
# Install espeak for TTS engine (optional, for fallback)
brew install espeak

# Or install via conda
conda install -c conda-forge espeak
```

### 5. Install Forced Alignment Tools (Optional)
```bash
# Install MFA for enhanced character alignment
cd backend
sh setup.sh

# Or manually install
conda config --add channels conda-forge
conda install montreal-forced-aligner
mfa model download acoustic english_us_arpa
mfa model download dictionary english_us_arpa
```

## 🚀 Usage

### **Starting the Server**
```bash
cd backend
python main.py
```

The server will start on `http://localhost:8001` with the default model (Chatterbox).

### **Testing Model Switching**
```bash
# Test the model switching functionality
python test_model_switching.py

# Run the demo
python demo_models.py
```

### **WebSocket TTS Usage**
```javascript
// Connect to TTS WebSocket
const ws = new WebSocket('ws://localhost:8001/ws/tts');

// Send text for TTS
ws.send(JSON.stringify({
    text: "Hello, this is a test!",
    flush: false
}));

// Force audio generation
ws.send(JSON.stringify({
    text: "",
    flush: true
}));

// Close connection
ws.send(JSON.stringify({
    text: "",
    flush: false
}));
```

### **Model Configuration Examples**

**Using Kokoro TTS:**
```python
# Switch to Kokoro
import requests
requests.post("http://localhost:8001/models/switch/kokoro")

# Kokoro will use:
# - Language code: 'a' (default)
# - Voice: 'af_heart' (default)
# - Sample rate: 24000 Hz (converted to 44.1 kHz output)
```

**Using Chatterbox TTS:**
```python
# Switch to Chatterbox
import requests
requests.post("http://localhost:8001/models/switch/chatterbox")

# Chatterbox will use:
# - High-quality streaming TTS
# - Configurable chunk sizes
# - GPU acceleration if available
```

### 6. Test Chatterbox TTS Integration
```bash
# Test the Chatterbox TTS integration
python test_chatterbox.py

# This will generate a test audio file to verify everything is working
```

### ⚡ Real-time Streaming with 20‑Word Sentence‑Aware Queueing

To deliver natural pacing with low latency, the backend maintains a per‑connection text buffer and enqueues chunks using a sentence‑aware 20‑word window:
- **Tokenization with Spaces Preserved**: We keep trailing whitespace to respect sentence boundaries.
- **Up to 20 Words per Chunk**: Prefer cutting on punctuation (., !, ?) within the window to avoid mid‑sentence breaks.
- **Immediate Enqueue**: Chunks are enqueued in an `asyncio.Queue` per connection as text arrives.
- **Concurrent Processing**: A background task pulls from the queue and generates audio, streaming results in real-time.
- **Flush Support**: Sending `{"text": "", "flush": true}` forces any remainder to be enqueued immediately.

Configuration:
- Set the chunk size in `config.py`:
```python
TTS_CONFIG = {
    # ...
    "chunk_word_count": 20,
}
```

Where it happens:
- The behavior is implemented in `backend/managers.py` inside `WebSocketManager.process_message` (sentence-aware chunking and queueing) and `process_audio_queue` (async consumption and audio generation).

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
5. Check the generated audio files in `backend/generated_audio/`

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

### Forced Alignment Settings
- **Primary Method**: Montreal Forced Aligner (MFA)
- **Fallback Method**: Generic proportional alignment
- **Alignment Type**: Character-level timing
- **Processing**: Real-time with audio generation

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
- `GET /models/current` - Current TTS model information
- `POST /models/switch/{model_type}` - Switch TTS models
- `GET /connections` - Active WebSocket connections

## 🔮 Future Enhancements

- **Advanced TTS Models**: Integration with neural TTS engines
- **Voice Selection**: Multiple voice options and customization
- **Audio Effects**: Pitch, speed, and tone adjustments
- **Streaming Optimization**: Adaptive chunk sizing and buffering
- **Authentication**: Secure WebSocket connections
- **Metrics Dashboard**: Real-time performance monitoring
- **Mobile Support**: Responsive design and touch controls
- **Enhanced Alignment**: Phoneme-level timing and emotion detection
- **Audio Post-processing**: Noise reduction and enhancement filters

## 📚 Technical Details

### Character Alignment Algorithm
The system now uses a sophisticated two-tier alignment approach:

#### **Tier 1: MFA Forced Alignment**
1. **Audio Processing**: Convert audio to MFA-compatible format
2. **Text Processing**: Prepare text for alignment
3. **MFA Execution**: Run Montreal Forced Aligner
4. **Output Parsing**: Extract word-level alignments from TextGrid
5. **Character Conversion**: Convert word timing to character timing

#### **Tier 2: Generic Fallback Alignment**
1. **Duration Calculation**: Calculate total audio duration
2. **Proportional Distribution**: Evenly distribute time across characters
3. **Timing Generation**: Generate start times and durations

### Audio Processing Pipeline
1. **Text Input** → TTS Engine (Chatterbox/Kokoro)
2. **Raw Audio Generation** → Native model format
3. **Raw File Save** → `{model}_{timestamp}_raw.wav`
4. **Format Conversion** → 44.1kHz, 16-bit, mono PCM
5. **Processed File Save** → `{model}_{timestamp}_processed.wav`
6. **Base64 Encoding** → WebSocket transmission
7. **Client Decoding** → Web Audio API processing
8. **Real-time Playback** → Synchronized with captions

### WebSocket Management
- **Connection Pooling**: Multiple concurrent connections
- **Message Buffering**: Efficient text accumulation
- **Error Handling**: Graceful failure recovery
- **Resource Cleanup**: Automatic connection management
- **Buffer Management**: Automatic clearing between requests

## 🧑‍💻 Technical Skills Gained

- **Real-time WebSocket development**: Designing bidirectional streaming APIs and clients.
- **Async Python with asyncio**: Queue-based producers/consumers, backpressure, and task orchestration.
- **Audio DSP in Python**: Resampling, dithering, DC offset removal, high‑pass filtering, edge fades, and level management.
- **TTS engine integration**: Wiring up Chatterbox and Kokoro models with standardized output.
- **Forced alignment tooling**: Installing, invoking, and parsing outputs from Montreal Forced Aligner.
- **Frontend Web Audio API**: Decoding PCM, buffer queuing, playback control, and UI state updates.
- **Node.js tooling for math speech**: Using MathJax + Speech Rule Engine via a Node helper.
- **Configuration and feature flags**: Runtime model switching and optional processing paths.
- **Diagnostics and troubleshooting**: Structured logging and validation of streaming pipelines.

## 📘 Technical Concepts Covered

- **Bidirectional streaming protocol design**: Chunking, flush semantics, and graceful termination.
- **Sentence‑aware text chunking**: 20‑word windowing with punctuation‑aware boundaries.
- **Forced alignment theory**: Word/phoneme alignment, TextGrid parsing, and char‑time projection.
- **PCM audio fundamentals**: Sample rate, bit depth, mono channelization, endianess, and normalization.
- **Polyphase resampling and windowing**: Quality impacts vs latency trade‑offs.
- **Dithering and noise shaping basics**: TPDF dither and quantization artifacts.
- **Math speech generation**: TeX → MathML → speech; ClearSpeak vs MathSpeak styles.
- **Symbol/units normalization**: Unicode operators and unit phrasing for intelligibility.
- **Transport considerations**: Base64 payload sizing and WebSocket delivery constraints.
- **Latency management**: Client buffering, playback scheduling, and concurrency impacts.

---

**Happy Text-to-Speech Streaming! 🎤✨**
