#!/bin/bash

# SigIQ TTS WebSocket System - Startup Script
# Updated for Dual TTS Model Support (Chatterbox + Kokoro)

echo "🎭 SigIQ TTS WebSocket System"
echo "🚀 Starting with Dual TTS Model Support"
echo "   - Chatterbox TTS: High-quality streaming TTS"
echo "   - Kokoro TTS: Lightweight, fast TTS (82M parameters)"
echo "=" * 50

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed or not in PATH"
    echo "💡 Please install Python 3.11+ and try again"
    exit 1
fi

# Check Python version
python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
required_version="3.11"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Python $python_version detected, but Python $required_version+ is required"
    echo "💡 Please upgrade Python and try again"
    exit 1
fi

echo "✅ Python $python_version detected"

# Check if backend directory exists
if [ ! -d "backend" ]; then
    echo "❌ Backend directory not found"
    echo "💡 Please run this script from the project root directory"
    exit 1
fi

# Check if requirements.txt exists
if [ ! -f "backend/requirements.txt" ]; then
    echo "❌ requirements.txt not found in backend directory"
    echo "💡 Please ensure the backend directory contains requirements.txt"
    exit 1
fi

# Install dependencies if needed
echo "📦 Checking dependencies..."
cd backend

# Check if TTS models are installed
if ! python3 -c "import chatterbox.tts" &> /dev/null || ! python3 -c "import kokoro" &> /dev/null; then
    echo "📥 Installing dependencies..."
    pip3 install -r requirements.txt
    
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install dependencies"
        echo "💡 Please check your Python environment and try again"
        exit 1
    fi
    echo "✅ Dependencies installed successfully"
else
    echo "✅ Dependencies already installed"
fi

# Test TTS model integration
echo "🧪 Testing TTS model integration..."
cd ..
if python3 test_chatterbox.py; then
    echo "✅ Chatterbox TTS integration test passed"
else
    echo "⚠️  Chatterbox TTS integration test failed, but continuing..."
fi

# Start the backend server
echo "🚀 Starting backend server..."
cd backend
echo "📍 Server will be available at: http://localhost:8001"
echo "🔌 WebSocket endpoint: ws://localhost:8001/ws/tts"
echo "📚 API documentation: http://localhost:8001/docs"
echo ""
echo "🎭 Model Management:"
echo "   GET  /models/current     - Show current TTS model"
echo "   GET  /models             - Show all available models"
echo "   POST /models/switch/kokoro     - Switch to Kokoro"
echo "   POST /models/switch/chatterbox - Switch to Chatterbox"
echo ""
echo "🧪 Test model switching: python3 test_model_switching.py"
echo "🎭 Interactive config: python3 config_models.py"
echo ""
echo "Press Ctrl+C to stop the server"
echo "=" * 50

python3 main.py
