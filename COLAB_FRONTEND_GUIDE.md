# 🚀 Google Colab TTS App + Frontend Deployment Guide

## 📋 Prerequisites
- Google Colab account (Colab Pro recommended for longer runtime)
- Your GitHub repository URL
- Basic knowledge of Python/Jupyter notebooks

## 🎯 Quick 3-Step Deployment

### **Step 1: Open Google Colab & Set Runtime**
1. Go to [Google Colab](https://colab.research.google.com/)
2. Create a new notebook
3. **CRITICAL**: Set runtime type to **GPU**
   - Runtime → Change runtime type → Hardware accelerator → GPU

### **Step 2: Clone & Deploy**
```python
# Clone your repository
!git clone https://github.com/YOUR_USERNAME/Text-to-Speech-realtime-app.git
!cd Text-to-Speech-realtime-app

# Run the deployment script
!python colab_tts_deploy.py
```

### **Step 3: Access Your App**
After the script completes, you'll get:
```
🎉 DEPLOYMENT COMPLETE!
🌐 Public URL: https://abc123.ngrok.io
🔗 Main App: https://abc123.ngrok.io (serves your index.html)
🔗 Frontend: https://abc123.ngrok.io/frontend
🔗 Test Interface: https://abc123.ngrok.io/test
```

## 🔧 What the Script Does Automatically

### **✅ Dependencies Installation**
- FastAPI, Uvicorn, WebSockets
- TTS engines (Chatterbox, Kokoro)
- Audio processing libraries
- Forced alignment tools

### **✅ Server Setup**
- Starts your `main.py` in background
- Serves `index.html` at root URL (/)
- Mounts static files from `/static/`
- Creates proper logging

### **✅ Ngrok Tunnel**
- Downloads and configures ngrok
- Creates HTTPS tunnel to port 8001
- Provides public access URL

### **✅ Frontend Serving**
- Your `index.html` is served at the root URL
- Static files accessible via `/static/`
- Test interface at `/test`
- API documentation at `/docs`

## 📱 How to Access Your App

### **Main Frontend (Your index.html)**
```
https://your-ngrok-url.ngrok.io/
```
This will automatically serve your `frontend/index.html` file!

### **Alternative Frontend Endpoints**
```
https://your-ngrok-url.ngrok.io/frontend
https://your-ngrok-url.ngrok.io/test
```

### **API Endpoints**
```
https://your-ngrok-url.ngrok.io/health
https://your-ngrok-url.ngrok.io/tts
https://your-ngrok-url.ngrok.io/docs
```

## 🧪 Testing Your Deployment

### **Test Frontend Access**
```python
# Test if your index.html is being served
!curl -s https://your-ngrok-url.ngrok.io/ | head -20
```

### **Test TTS Functionality**
```python
# Generate speech via API
!curl -X POST "https://your-ngrok-url.ngrok.io/tts" \
     -H "Content-Type: application/json" \
     -d '{"text": "Hello from Colab!", "model": "chatterbox"}' \
     --output test_audio.wav
```

### **Test WebSocket Connection**
```python
# Check WebSocket endpoint
!curl -s https://your-ngrok-url.ngrok.io/ | grep -i websocket
```

## 🔍 Troubleshooting

### **Frontend Not Loading**
```python
# Check server logs
!tail -50 server.log

# Check if index.html exists
!ls -la frontend/

# Check server health
!curl -s http://localhost:8001/health
```

### **Ngrok Issues**
```python
# Check ngrok logs
!tail -50 ngrok.log

# Restart ngrok
!pkill ngrok
!./ngrok http 8001 &
```

### **Server Issues**
```python
# Check if server is running
!ps aux | grep python

# Restart server
!pkill -f "python main.py"
!cd backend && python main.py &
```

## 🚨 Important Notes

### **Colab Limitations**
- **Runtime disconnects** after 12 hours (Pro: 24 hours)
- **GPU memory** limited to ~12GB
- **Background processes** may stop when notebook is idle

### **Ngrok Limitations**
- **Free tier**: Random URLs, limited bandwidth
- **URLs change** each time you restart ngrok
- **Public access** - anyone with URL can use your app

### **File Structure Requirements**
```
Text-to-Speech-realtime-app/
├── backend/
│   └── main.py          # Your TTS server
├── frontend/
│   └── index.html       # Your frontend
└── colab_tts_deploy.py  # Deployment script
```

## 🎉 Success Indicators

- [ ] All dependencies installed without errors
- [ ] TTS server running on port 8001
- [ ] Ngrok tunnel established
- [ ] Public URL accessible
- [ ] Your `index.html` loads at root URL
- [ ] TTS functionality working
- [ ] Frontend endpoints accessible

## 🚀 Next Steps After Deployment

1. **Test the public URL** in your browser
2. **Verify your index.html loads** correctly
3. **Test TTS functionality** via the web interface
4. **Share the URL** with others for testing
5. **Monitor performance** and resource usage
6. **Consider adding** authentication if needed

## 💡 Pro Tips

- **Keep the Colab notebook running** - server runs in background
- **Use Colab Pro** for longer runtime and better resources
- **Monitor GPU usage** in Colab's resource panel
- **The ngrok URL changes** each time you restart
- **Your frontend is automatically served** at the root URL

---

## 🎯 **You're Ready to Deploy!**

Just follow the 3 steps above:

1. **Open Colab + Set GPU runtime**
2. **Clone repo + Run deployment script**
3. **Access your app via the ngrok URL**

Your TTS app with frontend will be live on the internet in minutes! 🎵✨

The script handles everything automatically, so you can focus on using your app rather than fighting with deployment issues.
