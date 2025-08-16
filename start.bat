@echo off
REM SigIQ TTS System Startup Script for Windows

echo 🎤 Starting SigIQ TTS WebSocket System...
echo ==========================================

REM Check if conda environment exists
conda env list | findstr "tts" >nul
if %errorlevel% equ 0 (
    echo ✅ TTS conda environment found
    echo 🔄 Activating TTS environment...
    call conda activate tts
) else (
    echo ❌ TTS conda environment not found
    echo 📝 Creating TTS environment...
    conda create -n tts python=3.11.13 -y
    echo 🔄 Activating TTS environment...
    call conda activate tts
)

REM Check if dependencies are installed
echo 🔍 Checking dependencies...
python -c "import fastapi, torch, transformers, numpy" 2>nul
if %errorlevel% neq 0 (
    echo 📦 Installing dependencies...
    cd backend
    pip install -r requirements.txt
    cd ..
) else (
    echo ✅ Dependencies already installed
)

REM Start backend server
echo 🚀 Starting backend server...
cd backend
start /B python main.py
cd ..

REM Wait a moment for backend to start
timeout /t 3 /nobreak >nul

REM Check if backend is running
curl -s http://localhost:8000/health >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ Backend server started successfully on http://localhost:8000
) else (
    echo ❌ Backend server failed to start
    pause
    exit /b 1
)

REM Start frontend server
echo 🌐 Starting frontend server...
cd frontend
start /B python -m http.server 8080
cd ..

echo ✅ Frontend server started on http://localhost:8080
echo.
echo 🎉 SigIQ TTS System is now running!
echo.
echo 📱 Frontend: http://localhost:8080
echo 🔌 Backend: http://localhost:8000
echo 📚 API Docs: http://localhost:8000/docs
echo.
echo 🛑 To stop the system, close this window or press any key
echo.
pause >nul
