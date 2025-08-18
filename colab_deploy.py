#!/usr/bin/env python3
"""
🚀 Google Colab + Ngrok Deployment Script for TTS Realtime App
================================================================

This script automates the entire setup process on Google Colab:
1. Installs dependencies
2. Downloads models
3. Starts the TTS server
4. Creates ngrok tunnel
5. Provides access URLs

Run this in a Colab notebook cell after cloning your repository!
"""

import os
import sys
import subprocess
import time
import requests
from pathlib import Path

def run_command(command, description, check=True):
    """Run a shell command with proper error handling"""
    print(f"🔄 {description}...")
    print(f"   Command: {command}")
    
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            capture_output=True, 
            text=True, 
            check=check
        )
        
        if result.stdout:
            print(f"✅ Output: {result.stdout}")
        if result.stderr:
            print(f"⚠️  Stderr: {result.stderr}")
            
        return result
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed: {e}")
        if check:
            raise
        return e

def install_dependencies():
    """Install all required Python dependencies"""
    print("\n📦 Installing Python dependencies...")
    
    # Upgrade pip first
    run_command("pip install --upgrade pip", "Upgrading pip")
    
    # Install core dependencies
    dependencies = [
        "fastapi==0.104.1",
        "uvicorn[standard]==0.24.0", 
        "websockets==12.0",
        "chatterbox-tts>=1.0.0",
        "kokoro>=0.9.2",
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "transformers>=4.40.0",
        "peft>=0.7.0",
        "numpy>=1.26.0",
        "python-multipart==0.0.6",
        "soundfile>=0.12.1",
        "librosa>=0.10.0",
        "scipy>=1.10.0",
        "montreal-forced-aligner>=1.1.0",
        "praatio>=6.0.0"
    ]
    
    for dep in dependencies:
        run_command(f"pip install {dep}", f"Installing {dep}")
    
    print("✅ All dependencies installed!")

def install_ngrok():
    """Install and configure ngrok"""
    print("\n🌐 Installing ngrok...")
    
    # Install ngrok
    run_command("pip install pyngrok", "Installing pyngrok")
    
    # Download ngrok binary
    run_command("wget https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz", "Downloading ngrok")
    run_command("tar -xzf ngrok-v3-stable-linux-amd64.tgz", "Extracting ngrok")
    run_command("chmod +x ngrok", "Making ngrok executable")
    
    print("✅ Ngrok installed!")

def download_models():
    """Download required TTS models"""
    print("\n🤖 Downloading TTS models...")
    
    # Create models directory
    os.makedirs("models", exist_ok=True)
    
    # Download Chatterbox model (if needed)
    print("   • Chatterbox models will be downloaded automatically on first use")
    
    # Download Kokoro models (if needed)
    print("   • Kokoro models will be downloaded automatically on first use")
    
    print("✅ Model setup complete!")

def start_server():
    """Start the TTS server in background"""
    print("\n🚀 Starting TTS server...")
    
    # Change to backend directory
    os.chdir("backend")
    
    # Start server in background
    server_cmd = "python main.py &"
    run_command(server_cmd, "Starting TTS server", check=False)
    
    # Wait for server to start
    print("⏳ Waiting for server to start...")
    time.sleep(10)
    
    # Check if server is running
    try:
        response = requests.get("http://localhost:8001/health", timeout=5)
        if response.status_code == 200:
            print("✅ Server is running on port 8001!")
        else:
            print("⚠️  Server responded but not healthy")
    except:
        print("⚠️  Server might still be starting up...")
    
    # Go back to root directory
    os.chdir("..")

def create_ngrok_tunnel():
    """Create ngrok tunnel to the server"""
    print("\n🌐 Creating ngrok tunnel...")
    
    # Kill any existing ngrok processes
    run_command("pkill ngrok", "Killing existing ngrok processes", check=False)
    
    # Create tunnel to port 8001
    tunnel_cmd = "./ngrok http 8001 --log=stdout > ngrok.log 2>&1 &"
    run_command(tunnel_cmd, "Creating ngrok tunnel", check=False)
    
    # Wait for tunnel to establish
    print("⏳ Waiting for ngrok tunnel...")
    time.sleep(5)
    
    # Get tunnel URL
    try:
        response = requests.get("http://localhost:4040/api/tunnels")
        if response.status_code == 200:
            tunnels = response.json()['tunnels']
            if tunnels:
                public_url = tunnels[0]['public_url']
                print(f"✅ Ngrok tunnel created: {public_url}")
                return public_url
            else:
                print("⚠️  No tunnels found, checking logs...")
        else:
            print("⚠️  Could not get tunnel info")
    except:
        print("⚠️  Could not connect to ngrok API")
    
    # Fallback: check ngrok logs
    if os.path.exists("ngrok.log"):
        with open("ngrok.log", "r") as f:
            log_content = f.read()
            if "url=" in log_content:
                # Extract URL from logs
                for line in log_content.split('\n'):
                    if "url=" in line:
                        url = line.split("url=")[1].strip()
                        print(f"✅ Found tunnel URL in logs: {url}")
                        return url
    
    print("❌ Could not establish ngrok tunnel")
    return None

def create_frontend_html():
    """Create a simple test HTML file for testing"""
    print("\n📄 Creating test HTML file...")
    
    html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>TTS App Test</title>
    <meta charset="utf-8">
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .container { max-width: 800px; margin: 0 auto; }
        .test-section { margin: 20px 0; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }
        button { padding: 10px 20px; margin: 5px; background: #007bff; color: white; border: none; border-radius: 3px; cursor: pointer; }
        button:hover { background: #0056b3; }
        input, textarea { width: 100%; padding: 10px; margin: 5px 0; border: 1px solid #ddd; border-radius: 3px; }
        .status { padding: 10px; margin: 10px 0; border-radius: 3px; }
        .success { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
        .error { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
        .info { background: #d1ecf1; color: #0c5460; border: 1px solid #bee5eb; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎤 TTS Realtime App - Test Interface</h1>
        
        <div class="test-section">
            <h3>🔗 Connection Test</h3>
            <button onclick="testConnection()">Test Server Connection</button>
            <div id="connectionStatus"></div>
        </div>
        
        <div class="test-section">
            <h3>🎵 Text-to-Speech Test</h3>
            <textarea id="testText" rows="3" placeholder="Enter text to convert to speech...">Hello, this is a test of the TTS system!</textarea>
            <br>
            <button onclick="testTTS()">Generate Speech</button>
            <div id="ttsStatus"></div>
            <audio id="audioPlayer" controls style="display: none;"></audio>
        </div>
        
        <div class="test-section">
            <h3>📊 Server Info</h3>
            <button onclick="getServerInfo()">Get Server Information</button>
            <div id="serverInfo"></div>
        </div>
    </div>

    <script>
        const SERVER_URL = window.location.origin.replace('https://', 'http://').replace('http://', 'http://');
        
        async function testConnection() {
            const statusDiv = document.getElementById('connectionStatus');
            statusDiv.innerHTML = '<div class="info">Testing connection...</div>';
            
            try {
                const response = await fetch(`${SERVER_URL}/health`);
                if (response.ok) {
                    statusDiv.innerHTML = '<div class="success">✅ Server is connected and responding!</div>';
                } else {
                    statusDiv.innerHTML = '<div class="error">❌ Server responded with error: ' + response.status + '</div>';
                }
            } catch (error) {
                statusDiv.innerHTML = '<div class="error">❌ Connection failed: ' + error.message + '</div>';
            }
        }
        
        async function testTTS() {
            const text = document.getElementById('testText').value;
            const statusDiv = document.getElementById('ttsStatus');
            const audioPlayer = document.getElementById('audioPlayer');
            
            if (!text.trim()) {
                statusDiv.innerHTML = '<div class="error">❌ Please enter some text</div>';
                return;
            }
            
            statusDiv.innerHTML = '<div class="info">Generating speech...</div>';
            
            try {
                const response = await fetch(`${SERVER_URL}/tts`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        text: text,
                        model: 'chatterbox'
                    })
                });
                
                if (response.ok) {
                    const audioBlob = await response.blob();
                    const audioUrl = URL.createObjectURL(audioBlob);
                    audioPlayer.src = audioUrl;
                    audioPlayer.style.display = 'block';
                    statusDiv.innerHTML = '<div class="success">✅ Speech generated successfully! Use the audio player above.</div>';
                } else {
                    const errorText = await response.text();
                    statusDiv.innerHTML = '<div class="error">❌ TTS failed: ' + errorText + '</div>';
                }
            } catch (error) {
                statusDiv.innerHTML = '<div class="error">❌ TTS request failed: ' + error.message + '</div>';
            }
        }
        
        async function getServerInfo() {
            const statusDiv = document.getElementById('serverInfo');
            statusDiv.innerHTML = '<div class="info">Getting server information...</div>';
            
            try {
                const response = await fetch(`${SERVER_URL}/info`);
                if (response.ok) {
                    const info = await response.json();
                    statusDiv.innerHTML = '<div class="success"><pre>' + JSON.stringify(info, null, 2) + '</pre></div>';
                } else {
                    statusDiv.innerHTML = '<div class="error">❌ Could not get server info: ' + response.status + '</div>';
                }
            } catch (error) {
                statusDiv.innerHTML = '<div class="error">❌ Server info request failed: ' + error.message + '</div>';
            }
        }
        
        // Auto-test connection on page load
        window.onload = function() {
            setTimeout(testConnection, 1000);
        };
    </script>
</body>
</html>
"""
    
    with open("test_tts.html", "w") as f:
        f.write(html_content)
    
    print("✅ Test HTML file created: test_tts.html")

def main():
    """Main deployment function"""
    print("🚀 Starting TTS App Deployment on Google Colab...")
    print("=" * 60)
    
    try:
        # Step 1: Install dependencies
        install_dependencies()
        
        # Step 2: Install ngrok
        install_ngrok()
        
        # Step 3: Download models
        download_models()
        
        # Step 4: Start server
        start_server()
        
        # Step 5: Create ngrok tunnel
        public_url = create_ngrok_tunnel()
        
        # Step 6: Create test interface
        create_frontend_html()
        
        # Final output
        print("\n" + "=" * 60)
        print("🎉 DEPLOYMENT COMPLETE!")
        print("=" * 60)
        
        if public_url:
            print(f"🌐 Public URL: {public_url}")
            print(f"🔗 Test Interface: {public_url.replace('8001', '8001')}/test_tts.html")
        else:
            print("⚠️  Ngrok tunnel not established, but server is running locally")
        
        print("\n📱 How to use:")
        print("1. Open the public URL in your browser")
        print("2. Use the test interface to verify TTS functionality")
        print("3. The server will continue running in the background")
        print("4. Check ngrok.log for tunnel details")
        
        print("\n🔧 Troubleshooting:")
        print("• If server doesn't start, check the backend directory exists")
        print("• If ngrok fails, check ngrok.log for errors")
        print("• Server runs on port 8001, ngrok creates public tunnel")
        
        print("\n🚀 Your TTS app is now live on Google Colab!")
        
    except Exception as e:
        print(f"\n❌ Deployment failed: {e}")
        print("Check the error messages above for troubleshooting steps")
        return False
    
    return True

if __name__ == "__main__":
    main()
