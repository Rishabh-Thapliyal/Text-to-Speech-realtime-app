#!/bin/bash

# SigIQ TTS System Startup Script

echo "🎤 Starting SigIQ TTS WebSocket System..."
echo "=========================================="

# Check if conda environment exists
if conda env list | grep -q "tts"; then
    echo "✅ TTS conda environment found"
    echo "🔄 Activating TTS environment..."
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate tts
else
    echo "❌ TTS conda environment not found"
    echo "📝 Creating TTS environment..."
    conda create -n tts python=3.11.13 -y
    echo "🔄 Activating TTS environment..."
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate tts
fi

# Check if dependencies are installed
echo "🔍 Checking dependencies..."
if ! python -c "import fastapi, torch, transformers, numpy" 2>/dev/null; then
    echo "📦 Installing dependencies..."
    cd backend
    pip install -r requirements.txt
    cd ..
else
    echo "✅ Dependencies already installed"
fi

# Start backend server
echo "🚀 Starting backend server..."
cd backend
python main.py &
BACKEND_PID=$!
cd ..

# Wait a moment for backend to start
sleep 3

# Check if backend is running
if curl -s http://localhost:8000/health > /dev/null; then
    echo "✅ Backend server started successfully on http://localhost:8000"
else
    echo "❌ Backend server failed to start"
    exit 1
fi

# Start frontend server
echo "🌐 Starting frontend server..."
cd frontend
python -m http.server 8080 &
FRONTEND_PID=$!
cd ..

echo "✅ Frontend server started on http://localhost:8080"
echo ""
echo "🎉 SigIQ TTS System is now running!"
echo ""
echo "📱 Frontend: http://localhost:8080"
echo "🔌 Backend: http://localhost:8000"
echo "📚 API Docs: http://localhost:8000/docs"
echo ""
echo "🛑 To stop the system, press Ctrl+C"
echo ""

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Shutting down SigIQ TTS System..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    echo "✅ System stopped"
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Keep script running
wait
