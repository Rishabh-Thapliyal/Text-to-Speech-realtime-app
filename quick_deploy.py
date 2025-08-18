#!/usr/bin/env python3
"""
🚀 Quick Deploy Script for Google Colab
========================================

This is a simplified version that does everything in one go.
Just run this script after cloning your repository!
"""

import os, subprocess, time, requests

print("🚀 Quick Deploy Starting...")

# Install dependencies
print("📦 Installing dependencies...")
subprocess.run("pip install --upgrade pip", shell=True)
subprocess.run("pip install fastapi uvicorn websockets chatterbox-tts kokoro torch torchaudio transformers numpy soundfile librosa scipy", shell=True)

# Install ngrok
print("🌐 Installing ngrok...")
subprocess.run("pip install pyngrok", shell=True)
subprocess.run("wget https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz", shell=True)
subprocess.run("tar -xzf ngrok-v3-stable-linux-amd64.tgz", shell=True)
subprocess.run("chmod +x ngrok", shell=True)

# Start server
print("🚀 Starting TTS server...")
os.chdir("backend")
subprocess.run("python main.py &", shell=True)
os.chdir("..")
time.sleep(10)

# Create ngrok tunnel
print("🌐 Creating ngrok tunnel...")
subprocess.run("pkill ngrok", shell=True)
subprocess.run("./ngrok http 8001 --log=stdout > ngrok.log 2>&1 &", shell=True)
time.sleep(5)

# Get tunnel URL
try:
    response = requests.get("http://localhost:4040/api/tunnels")
    if response.status_code == 200:
        tunnels = response.json()['tunnels']
        if tunnels:
            public_url = tunnels[0]['public_url']
            print(f"\n🎉 DEPLOYMENT COMPLETE!")
            print(f"🌐 Public URL: {public_url}")
            print(f"🔗 Test: {public_url}/test_tts.html")
        else:
            print("⚠️ Check ngrok.log for tunnel details")
    else:
        print("⚠️ Could not get tunnel info")
except:
    print("⚠️ Check ngrok.log for tunnel details")

print("\n🚀 Your TTS app is now live on Google Colab!")
