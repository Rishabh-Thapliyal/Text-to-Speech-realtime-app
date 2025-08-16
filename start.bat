@echo off
REM SigIQ TTS WebSocket System - Startup Script for Windows
REM Updated for Chatterbox TTS Integration

echo 🎤 SigIQ TTS WebSocket System
echo 🚀 Starting with Chatterbox TTS Integration
echo ==================================================

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo 💡 Please install Python 3.11+ and try again
    pause
    exit /b 1
)

echo ✅ Python detected

REM Check if backend directory exists
if not exist "backend" (
    echo ❌ Backend directory not found
    echo 💡 Please run this script from the project root directory
    pause
    exit /b 1
)

REM Check if requirements.txt exists
if not exist "backend\requirements.txt" (
    echo ❌ requirements.txt not found in backend directory
    echo 💡 Please ensure the backend directory contains requirements.txt
    pause
    exit /b 1
)

REM Install dependencies if needed
echo 📦 Checking dependencies...
cd backend

REM Check if chatterbox-tts is installed
python -c "import chatterbox.tts" >nul 2>&1
if errorlevel 1 (
    echo 📥 Installing dependencies...
    pip install -r requirements.txt
    
    if errorlevel 1 (
        echo ❌ Failed to install dependencies
        echo 💡 Please check your Python environment and try again
        pause
        exit /b 1
    )
    echo ✅ Dependencies installed successfully
) else (
    echo ✅ Dependencies already installed
)

REM Test Chatterbox TTS integration
echo 🧪 Testing Chatterbox TTS integration...
cd ..
python test_chatterbox.py
if errorlevel 1 (
    echo ⚠️  Chatterbox TTS integration test failed, but continuing...
) else (
    echo ✅ Chatterbox TTS integration test passed
)

REM Start the backend server
echo 🚀 Starting backend server...
cd backend
echo 📍 Server will be available at: http://localhost:8000
echo 🔌 WebSocket endpoint: ws://localhost:8000/ws/tts
echo 📚 API documentation: http://localhost:8000/docs
echo.
echo Press Ctrl+C to stop the server
echo ==================================================

python main.py
