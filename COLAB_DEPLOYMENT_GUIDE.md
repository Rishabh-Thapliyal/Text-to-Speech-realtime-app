# 🚀 Google Colab + Ngrok Deployment Guide

## 📋 Prerequisites
- Google Colab account (Colab Pro recommended for longer runtime)
- Your GitHub repository URL
- Basic knowledge of Python/Jupyter notebooks

## 🎯 Step-by-Step Deployment

### **Step 1: Open Google Colab**
1. Go to [Google Colab](https://colab.research.google.com/)
2. Create a new notebook
3. **IMPORTANT**: Set runtime type to **GPU** for better performance
   - Runtime → Change runtime type → Hardware accelerator → GPU

### **Step 2: Clone Your Repository**
```python
# Replace with your actual GitHub details
!git clone https://github.com/YOUR_USERNAME/Text-to-Speech-realtime-app.git
!cd Text-to-Speech-realtime-app
!ls -la
```

### **Step 3: Run the Deployment Script**
```python
# Make sure you're in the right directory
%cd Text-to-Speech-realtime-app

# Run the deployment script
!python colab_deploy.py
```

### **Step 4: Wait for Completion**
The script will automatically:
- ✅ Install all Python dependencies
- ✅ Download ngrok
- ✅ Start the TTS server
- ✅ Create ngrok tunnel
- ✅ Provide public access URL

### **Step 5: Get Your Public URL**
After the script completes, you'll see output like:
```
🎉 DEPLOYMENT COMPLETE!
🌐 Public URL: https://abc123.ngrok.io
🔗 Test Interface: https://abc123.ngrok.io/test_tts.html
```

## 🔧 Manual Commands (if needed)

### **Install Dependencies Manually**
```python
# Upgrade pip
!pip install --upgrade pip

# Install core dependencies
!pip install fastapi==0.104.1 uvicorn[standard]==0.24.0 websockets==12.0

# Install TTS engines
!pip install chatterbox-tts>=1.0.0 kokoro>=0.9.2

# Install audio processing
!pip install numpy soundfile librosa scipy

# Install forced alignment
!pip install montreal-forced-aligner praatio
```

### **Install Ngrok Manually**
```python
# Install pyngrok
!pip install pyngrok

# Download ngrok binary
!wget https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz
!tar -xzf ngrok-v3-stable-linux-amd64.tgz
!chmod +x ngrok
```

### **Start Server Manually**
```python
# Start server in background
!cd backend && python main.py &

# Wait for server to start
import time
time.sleep(10)

# Check if server is running
!curl -s http://localhost:8001/health
```

### **Create Ngrok Tunnel Manually**
```python
# Kill existing ngrok processes
!pkill ngrok

# Create tunnel
!./ngrok http 8001 --log=stdout > ngrok.log 2>&1 &

# Wait for tunnel
import time
time.sleep(5)

# Get tunnel URL
!curl -s http://localhost:4040/api/tunnels | python -m json.tool
```

## 🧪 Testing Your Deployment

### **Test Server Health**
```python
!curl -s http://localhost:8001/health
```

### **Test TTS Functionality**
```python
!curl -X POST "http://localhost:8001/tts" \
     -H "Content-Type: application/json" \
     -d '{"text": "Hello from Google Colab!", "model": "chatterbox"}' \
     --output test_audio.wav

print("✅ Audio file generated!")
```

### **Access Test Interface**
1. Open your ngrok URL in a browser
2. Navigate to `/test_tts.html`
3. Use the web interface to test TTS

## 🔍 Troubleshooting

### **Common Issues & Solutions**

#### **1. Server Won't Start**
```python
# Check if backend directory exists
!ls -la backend/

# Check Python version
!python --version

# Check if main.py exists
!ls -la backend/main.py
```

#### **2. Ngrok Tunnel Fails**
```python
# Check ngrok logs
!tail -50 ngrok.log

# Check if ngrok is running
!ps aux | grep ngrok

# Restart ngrok
!pkill ngrok
!./ngrok http 8001 &
```

#### **3. Dependencies Installation Fails**
```python
# Clear pip cache
!pip cache purge

# Install with no cache
!pip install --no-cache-dir chatterbox-tts

# Check what's installed
!pip list | grep chatterbox
```

#### **4. Port Already in Use**
```python
# Check what's using port 8001
!lsof -i :8001

# Kill processes using the port
!pkill -f "python main.py"
```

### **Resource Monitoring**
```python
# Check GPU usage
!nvidia-smi

# Check memory usage
!free -h

# Check disk space
!df -h
```

## 📱 Using Your Deployed App

### **API Endpoints**
- **Health Check**: `GET /health`
- **TTS Generation**: `POST /tts`
- **Server Info**: `GET /info`
- **WebSocket**: `ws://your-ngrok-url/ws`

### **Example API Usage**
```python
import requests

# Generate speech
response = requests.post(
    "https://your-ngrok-url/tts",
    json={
        "text": "Hello, this is a test!",
        "model": "chatterbox"
    }
)

if response.status_code == 200:
    with open("output.wav", "wb") as f:
        f.write(response.content)
    print("✅ Audio generated!")
```

### **WebSocket Usage**
```javascript
const ws = new WebSocket('wss://your-ngrok-url/ws');

ws.onopen = function() {
    console.log('Connected to TTS server');
    ws.send(JSON.stringify({
        text: "Hello via WebSocket!",
        model: "chatterbox"
    }));
};

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    console.log('Received:', data);
};
```

## 🚨 Important Notes

### **Colab Limitations**
- **Runtime disconnects** after 12 hours (Pro: 24 hours)
- **GPU memory** limited to ~12GB
- **Disk space** limited to ~100GB
- **Background processes** may stop when notebook is idle

### **Ngrok Limitations**
- **Free tier**: 1 tunnel, random URLs, limited bandwidth
- **Paid tier**: Custom domains, more bandwidth, better reliability
- **URLs change** each time you restart ngrok

### **Security Considerations**
- **Public access** - anyone with the URL can use your app
- **No authentication** by default
- **Resource usage** visible to others
- **Consider adding** rate limiting and authentication

## 🎉 Success Checklist

- [ ] Repository cloned successfully
- [ ] All dependencies installed
- [ ] TTS server running on port 8001
- [ ] Ngrok tunnel established
- [ ] Public URL accessible
- [ ] TTS functionality working
- [ ] Test interface accessible

## 🆘 Getting Help

If you encounter issues:

1. **Check the logs** (server logs, ngrok logs)
2. **Verify all steps** were completed
3. **Check Colab resources** (GPU, memory, disk)
4. **Restart the process** if needed
5. **Use the troubleshooting** commands above

## 🚀 Next Steps

Once deployed successfully:

1. **Test all functionality** thoroughly
2. **Share the URL** with others for testing
3. **Monitor performance** and resource usage
4. **Consider adding** authentication and rate limiting
5. **Backup your configuration** and models

---

**Happy Deploying! 🎵✨**

Your TTS app is now accessible from anywhere on the internet via Google Colab!
