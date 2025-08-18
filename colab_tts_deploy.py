#!/usr/bin/env python3
"""
🚀 Google Colab TTS App Deployment Script
==========================================

This script automatically:
1. Installs all dependencies
2. Sets up ngrok
3. Starts the TTS server (main.py)
4. Serves the frontend (index.html)
5. Creates ngrok tunnel
6. Provides access URLs

Run this in a Colab notebook cell after cloning your repository!
"""

import os
import sys
import subprocess
import time
import requests
import json
from pathlib import Path

def run_command(command, description, check=True, shell=True):
    """Run a shell command with proper error handling"""
    print(f"🔄 {description}...")
    print(f"   Command: {command}")
    
    try:
        if shell:
            result = subprocess.run(command, shell=True, capture_output=True, text=True, check=check)
        else:
            result = subprocess.run(command, capture_output=True, text=True, check=check)
        
        if result.stdout:
            print(f"✅ Output: {result.stdout.strip()}")
        if result.stderr:
            print(f"⚠️  Stderr: {result.stderr.strip()}")
            
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
    
    # Install pyngrok
    run_command("pip install pyngrok", "Installing pyngrok")
    
    # Download ngrok binary
    run_command("wget https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz", "Downloading ngrok")
    run_command("tar -xzf ngrok-v3-stable-linux-amd64.tgz", "Extracting ngrok")
    run_command("chmod +x ngrok", "Making ngrok executable")
    
    print("✅ Ngrok installed!")

def check_file_structure():
    """Check if required files exist"""
    print("\n📁 Checking file structure...")
    
    required_files = [
        "backend/main.py",
        "frontend/index.html"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
            print(f"❌ Missing: {file_path}")
        else:
            print(f"✅ Found: {file_path}")
            # Show file size and permissions
            stat = os.stat(file_path)
            print(f"   Size: {stat.st_size} bytes, Permissions: {oct(stat.st_mode)[-3:]}")
    
    # Check current working directory
    print(f"\n📍 Current working directory: {os.getcwd()}")
    
    # List all files in current directory
    print("📂 Files in current directory:")
    for item in os.listdir("."):
        if os.path.isdir(item):
            print(f"   📁 {item}/")
        else:
            print(f"   📄 {item}")
    
    # List files in frontend directory
    if os.path.exists("frontend"):
        print("\n📂 Files in frontend directory:")
        for item in os.listdir("frontend"):
            if os.path.isdir(item):
                print(f"   📁 {item}/")
            else:
                print(f"   📄 {item}")
    
    if missing_files:
        print(f"\n⚠️  Missing files: {missing_files}")
        print("Please make sure you're in the correct directory and all files are present.")
        return False
    
    print("✅ All required files found!")
    return True

def start_tts_server():
    """Start the TTS server in background"""
    print("\n🚀 Starting TTS server...")
    
    # Kill any existing Python processes on port 8001
    run_command("pkill -f 'python.*main.py'", "Killing existing TTS server processes", check=False)
    run_command("pkill -f 'uvicorn.*main:app'", "Killing existing uvicorn processes", check=False)
    
    # Wait a moment for processes to stop
    time.sleep(2)
    
    # Change to backend directory and start server
    os.chdir("backend")
    
    # Start server in background with proper logging
    server_cmd = "python main.py > ../server.log 2>&1 &"
    run_command(server_cmd, "Starting TTS server in background", check=False)
    
    # Wait for server to start
    print("⏳ Waiting for server to start...")
    time.sleep(15)
    
    # Check if server is running
    try:
        response = requests.get("http://localhost:8001/health", timeout=10)
        if response.status_code == 200:
            print("✅ TTS server is running on port 8001!")
            server_info = response.json()
            print(f"   Status: {server_info.get('status', 'unknown')}")
            print(f"   TTS Engine: {server_info.get('tts_engine', {}).get('model_type', 'unknown')}")
        else:
            print(f"⚠️  Server responded but not healthy: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"⚠️  Server might still be starting up: {e}")
        print("   Check server.log for details")
    
    # Go back to root directory
    os.chdir("..")
    
    return True

def restart_tts_server():
    """Restart the TTS server to ensure proper configuration"""
    print("\n🔄 Restarting TTS server with updated configuration...")
    
    # Kill any existing Python processes on port 8001
    run_command("pkill -f 'python.*main.py'", "Killing existing TTS server processes", check=False)
    run_command("pkill -f 'uvicorn.*main:app'", "Killing existing uvicorn processes", check=False)
    
    # Wait for processes to stop
    time.sleep(3)
    
    # Change to backend directory and start server
    os.chdir("backend")
    
    # Start server in background with proper logging
    server_cmd = "python main.py > ../server.log 2>&1 &"
    run_command(server_cmd, "Starting TTS server in background", check=False)
    
    # Wait for server to start
    print("⏳ Waiting for server to start...")
    time.sleep(20)  # Increased wait time
    
    # Check if server is running
    max_attempts = 5
    for attempt in range(max_attempts):
        try:
            response = requests.get("http://localhost:8001/health", timeout=10)
            if response.status_code == 200:
                print("✅ TTS server is running on port 8001!")
                server_info = response.json()
                print(f"   Status: {server_info.get('status', 'unknown')}")
                print(f"   TTS Engine: {server_info.get('tts_engine', {}).get('model_type', 'unknown')}")
                
                # Test if frontend files are accessible
                try:
                    frontend_response = requests.get("http://localhost:8001/", timeout=5)
                    if frontend_response.status_code == 200:
                        print("✅ Frontend is accessible at root URL!")
                    else:
                        print(f"⚠️  Frontend not accessible: {frontend_response.status_code}")
                except:
                    print("⚠️  Frontend test failed")
                
                break
            else:
                print(f"⚠️  Server responded but not healthy: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"⚠️  Attempt {attempt + 1}/{max_attempts}: Server might still be starting up: {e}")
            if attempt < max_attempts - 1:
                print("   Waiting 5 more seconds...")
                time.sleep(5)
            else:
                print("   Server failed to start after multiple attempts")
                print("   Check server.log for details")
                return False
    
    # Go back to root directory
    os.chdir("..")
    
    return True

def create_ngrok_tunnel():
    """Create ngrok tunnel to the server"""
    print("\n🌐 Creating ngrok tunnel...")
    
    # Kill any existing ngrok processes
    run_command("pkill ngrok", "Killing existing ngrok processes", check=False)
    time.sleep(2)
    
    # Create tunnel to port 8001
    tunnel_cmd = "./ngrok http 8001 --log=stdout > ngrok.log 2>&1 &"
    run_command(tunnel_cmd, "Creating ngrok tunnel", check=False)
    
    # Wait for tunnel to establish
    print("⏳ Waiting for ngrok tunnel...")
    time.sleep(8)
    
    # Get tunnel URL
    public_url = None
    try:
        response = requests.get("http://localhost:4040/api/tunnels", timeout=5)
        if response.status_code == 200:
            tunnels = response.json()['tunnels']
            if tunnels:
                public_url = tunnels[0]['public_url']
                print(f"✅ Ngrok tunnel created: {public_url}")
            else:
                print("⚠️  No tunnels found, checking logs...")
        else:
            print("⚠️  Could not get tunnel info")
    except Exception as e:
        print(f"⚠️  Could not connect to ngrok API: {e}")
    
    # Fallback: check ngrok logs
    if not public_url and os.path.exists("ngrok.log"):
        with open("ngrok.log", "r") as f:
            log_content = f.read()
            if "url=" in log_content:
                # Extract URL from logs
                for line in log_content.split('\n'):
                    if "url=" in line:
                        url = line.split("url=")[1].strip()
                        if url.startswith("http"):
                            public_url = url
                            print(f"✅ Found tunnel URL in logs: {public_url}")
                            break
    
    if not public_url:
        print("❌ Could not establish ngrok tunnel")
        print("   Check ngrok.log for errors")
        return None
    
    return public_url

def test_frontend_access(public_url):
    """Test if frontend is accessible via ngrok"""
    print("\n🧪 Testing frontend access...")
    
    try:
        # Test root endpoint (should serve index.html)
        response = requests.get(public_url, timeout=10)
        if response.status_code == 200:
            content = response.text
            if "<html" in content.lower():
                print("✅ Frontend (index.html) is accessible via ngrok!")
                print(f"   URL: {public_url}")
            else:
                print("⚠️  Root endpoint accessible but doesn't contain HTML")
                print(f"   Response: {content[:200]}...")
        else:
            print(f"❌ Root endpoint not accessible: {response.status_code}")
            
        # Test frontend endpoint
        frontend_url = f"{public_url}/frontend"
        response = requests.get(frontend_url, timeout=10)
        if response.status_code == 200:
            print(f"✅ Frontend endpoint accessible: {frontend_url}")
        else:
            print(f"⚠️  Frontend endpoint not accessible: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error testing frontend access: {e}")
    
    return True

def create_test_interface():
    """Create a simple test HTML file for testing"""
    print("\n📄 Creating test HTML file...")
    
    html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>TTS App Test</title>
    <meta charset="utf-8">
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
        .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .test-section { margin: 20px 0; padding: 20px; border: 1px solid #ddd; border-radius: 5px; background: #fafafa; }
        button { padding: 12px 24px; margin: 8px; background: #007bff; color: white; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; }
        button:hover { background: #0056b3; }
        input, textarea { width: 100%; padding: 12px; margin: 8px 0; border: 1px solid #ddd; border-radius: 5px; font-size: 16px; }
        .status { padding: 15px; margin: 15px 0; border-radius: 5px; font-weight: bold; }
        .success { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
        .error { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
        .info { background: #d1ecf1; color: #0c5460; border: 1px solid #bee5eb; }
        h1 { color: #333; text-align: center; }
        h3 { color: #555; border-bottom: 2px solid #007bff; padding-bottom: 10px; }
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
            <textarea id="testText" rows="4" placeholder="Enter text to convert to speech...">Hello, this is a test of the TTS system running on Google Colab!</textarea>
            <br>
            <button onclick="testTTS()">Generate Speech</button>
            <div id="ttsStatus"></div>
            <audio id="audioPlayer" controls style="display: none; margin-top: 15px;"></audio>
        </div>
        
        <div class="test-section">
            <h3>📊 Server Info</h3>
            <button onclick="getServerInfo()">Get Server Information</button>
            <div id="serverInfo"></div>
        </div>
        
        <div class="test-section">
            <h3>🌐 Access URLs</h3>
            <p><strong>Main App:</strong> <a href="/" target="_blank">Root URL</a></p>
            <p><strong>Frontend:</strong> <a href="/frontend" target="_blank">Frontend Endpoint</a></p>
            <p><strong>API Docs:</strong> <a href="/docs" target="_blank">Interactive API Documentation</a></p>
        </div>
    </div>

    <script>
        const SERVER_URL = window.location.origin;
        
        async function testConnection() {
            const statusDiv = document.getElementById('connectionStatus');
            statusDiv.innerHTML = '<div class="info">Testing connection...</div>';
            
            try {
                const response = await fetch(`${SERVER_URL}/health`);
                if (response.ok) {
                    const data = await response.json();
                    statusDiv.innerHTML = `<div class="success">✅ Server is connected and responding!<br>Status: ${data.status}<br>TTS Engine: ${data.tts_engine?.model_type || 'Unknown'}</div>`;
                } else {
                    statusDiv.innerHTML = `<div class="error">❌ Server responded with error: ${response.status}</div>`;
                }
            } catch (error) {
                statusDiv.innerHTML = `<div class="error">❌ Connection failed: ${error.message}</div>`;
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
                    statusDiv.innerHTML = `<div class="error">❌ TTS failed: ${errorText}</div>`;
                }
            } catch (error) {
                statusDiv.innerHTML = `<div class="error">❌ TTS request failed: ${error.message}</div>`;
            }
        }
        
        async function getServerInfo() {
            const statusDiv = document.getElementById('serverInfo');
            statusDiv.innerHTML = '<div class="info">Getting server information...</div>';
            
            try {
                const response = await fetch(`${SERVER_URL}/models`);
                if (response.ok) {
                    const info = await response.json();
                    statusDiv.innerHTML = `<div class="success"><pre>${JSON.stringify(info, null, 2)}</pre></div>`;
                } else {
                    statusDiv.innerHTML = `<div class="error">❌ Could not get server info: ${response.status}</div>`;
                }
            } catch (error) {
                statusDiv.innerHTML = `<div class="error">❌ Server info request failed: ${error.message}</div>`;
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

def create_simple_frontend():
    """Create a simple frontend HTML file for testing"""
    print("\n📄 Creating simple frontend HTML file...")
    
    # Create frontend directory if it doesn't exist
    os.makedirs("frontend", exist_ok=True)
    
    html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>TTS Realtime App</title>
    <meta charset="utf-8">
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background: #f0f2f5; }
        .container { max-width: 900px; margin: 0 auto; background: white; padding: 30px; border-radius: 15px; box-shadow: 0 4px 20px rgba(0,0,0,0.1); }
        .header { text-align: center; margin-bottom: 30px; }
        .header h1 { color: #1a73e8; margin-bottom: 10px; }
        .header p { color: #5f6368; font-size: 18px; }
        .test-section { margin: 25px 0; padding: 25px; border: 2px solid #e8eaed; border-radius: 10px; background: #fafbfc; }
        .test-section h3 { color: #202124; margin-top: 0; border-bottom: 2px solid #1a73e8; padding-bottom: 10px; }
        button { padding: 15px 30px; margin: 10px 5px; background: #1a73e8; color: white; border: none; border-radius: 8px; cursor: pointer; font-size: 16px; font-weight: 500; transition: background 0.3s; }
        button:hover { background: #1557b0; }
        button:active { transform: translateY(1px); }
        input, textarea { width: 100%; padding: 15px; margin: 10px 0; border: 2px solid #e8eaed; border-radius: 8px; font-size: 16px; box-sizing: border-box; }
        input:focus, textarea:focus { outline: none; border-color: #1a73e8; box-shadow: 0 0 0 3px rgba(26,115,232,0.1); }
        .status { padding: 15px; margin: 15px 0; border-radius: 8px; font-weight: 500; }
        .success { background: #e6f4ea; color: #137333; border: 1px solid #34a853; }
        .error { background: #fce8e6; color: #c5221f; border: 1px solid #ea4335; }
        .info { background: #e8f0fe; color: #174ea6; border: 1px solid #4285f4; }
        .audio-player { margin-top: 20px; width: 100%; }
        .url-links { display: flex; gap: 15px; flex-wrap: wrap; margin-top: 15px; }
        .url-links a { padding: 10px 20px; background: #f1f3f4; color: #1a73e8; text-decoration: none; border-radius: 6px; font-weight: 500; transition: background 0.3s; }
        .url-links a:hover { background: #e8eaed; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎤 TTS Realtime App</h1>
            <p>Text-to-Speech Application Running on Google Colab</p>
        </div>
        
        <div class="test-section">
            <h3>🔗 Connection Test</h3>
            <button onclick="testConnection()">Test Server Connection</button>
            <div id="connectionStatus"></div>
        </div>
        
        <div class="test-section">
            <h3>🎵 Text-to-Speech Test</h3>
            <textarea id="testText" rows="4" placeholder="Enter text to convert to speech...">Hello, this is a test of the TTS system running on Google Colab!</textarea>
            <br>
            <button onclick="testTTS()">Generate Speech</button>
            <div id="ttsStatus"></div>
            <audio id="audioPlayer" controls class="audio-player" style="display: none;"></audio>
        </div>
        
        <div class="test-section">
            <h3>📊 Server Information</h3>
            <button onclick="getServerInfo()">Get Server Information</button>
            <div id="serverInfo"></div>
        </div>
        
        <div class="test-section">
            <h3>🌐 Access URLs</h3>
            <p>Use these links to access different parts of your app:</p>
            <div class="url-links">
                <a href="/" target="_blank">🏠 Main App (Root)</a>
                <a href="/frontend" target="_blank">📱 Frontend Endpoint</a>
                <a href="/test" target="_blank">🧪 Test Interface</a>
                <a href="/docs" target="_blank">📚 API Documentation</a>
                <a href="/health" target="_blank">💚 Health Check</a>
            </div>
        </div>
        
        <div class="test-section">
            <h3>🔧 WebSocket Test</h3>
            <button onclick="testWebSocket()">Test WebSocket Connection</button>
            <div id="websocketStatus"></div>
        </div>
    </div>

    <script>
        const SERVER_URL = window.location.origin;
        
        async function testConnection() {
            const statusDiv = document.getElementById('connectionStatus');
            statusDiv.innerHTML = '<div class="info">Testing connection...</div>';
            
            try {
                const response = await fetch(`${SERVER_URL}/health`);
                if (response.ok) {
                    const data = await response.json();
                    statusDiv.innerHTML = `<div class="success">✅ Server is connected and responding!<br><strong>Status:</strong> ${data.status}<br><strong>TTS Engine:</strong> ${data.tts_engine?.model_type || 'Unknown'}<br><strong>Active Connections:</strong> ${data.active_connections || 0}</div>`;
                } else {
                    statusDiv.innerHTML = `<div class="error">❌ Server responded with error: ${response.status}</div>`;
                }
            } catch (error) {
                statusDiv.innerHTML = `<div class="error">❌ Connection failed: ${error.message}</div>`;
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
                    statusDiv.innerHTML = `<div class="error">❌ TTS failed: ${errorText}</div>`;
                }
            } catch (error) {
                statusDiv.innerHTML = `<div class="error">❌ TTS request failed: ${error.message}</div>`;
            }
        }
        
        async function getServerInfo() {
            const statusDiv = document.getElementById('serverInfo');
            statusDiv.innerHTML = '<div class="info">Getting server information...</div>';
            
            try {
                const response = await fetch(`${SERVER_URL}/models`);
                if (response.ok) {
                    const info = await response.json();
                    statusDiv.innerHTML = `<div class="success"><pre>${JSON.stringify(info, null, 2)}</pre></div>`;
                } else {
                    statusDiv.innerHTML = `<div class="error">❌ Could not get server info: ${response.status}</div>`;
                }
            } catch (error) {
                statusDiv.innerHTML = `<div class="error">❌ Server info request failed: ${error.message}</div>`;
            }
        }
        
        function testWebSocket() {
            const statusDiv = document.getElementById('websocketStatus');
            statusDiv.innerHTML = '<div class="info">Testing WebSocket connection...</div>';
            
            try {
                const ws = new WebSocket(`ws${window.location.protocol === 'https:' ? 's' : ''}://${window.location.host}/ws/tts`);
                
                ws.onopen = function() {
                    statusDiv.innerHTML = '<div class="success">✅ WebSocket connection established!</div>';
                    ws.close();
                };
                
                ws.onerror = function(error) {
                    statusDiv.innerHTML = '<div class="error">❌ WebSocket connection failed</div>';
                };
                
                setTimeout(() => {
                    if (ws.readyState === WebSocket.CONNECTING) {
                        statusDiv.innerHTML = '<div class="error">❌ WebSocket connection timeout</div>';
                        ws.close();
                    }
                }, 5000);
                
            } catch (error) {
                statusDiv.innerHTML = `<div class="error">❌ WebSocket test failed: ${error.message}</div>`;
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
    
    with open("frontend/index.html", "w") as f:
        f.write(html_content)
    
    print("✅ Simple frontend HTML file created: frontend/index.html")
    print(f"   File size: {os.path.getsize('frontend/index.html')} bytes")
    print(f"   Full path: {os.path.abspath('frontend/index.html')}")

def verify_frontend_files():
    """Verify that frontend files exist and are accessible"""
    print("\n🔍 Verifying frontend files...")
    
    required_files = [
        "frontend/index.html",
        "test_tts.html"
    ]
    
    all_exist = True
    for file_path in required_files:
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            print(f"✅ {file_path} - {file_size} bytes")
            
            # Try to read the file to ensure it's accessible
            try:
                with open(file_path, 'r') as f:
                    content = f.read(100)  # Read first 100 characters
                    if "<html" in content.lower():
                        print(f"   ✓ Valid HTML file")
                    else:
                        print(f"   ⚠️  File exists but may not be HTML")
            except Exception as e:
                print(f"   ❌ Error reading file: {e}")
                all_exist = False
        else:
            print(f"❌ {file_path} - Missing!")
            all_exist = False
    
    if all_exist:
        print("✅ All frontend files verified and accessible!")
        return True
    else:
        print("❌ Some frontend files are missing or inaccessible!")
        return False

def main():
    """Main deployment function"""
    print("🚀 Starting TTS App Deployment on Google Colab...")
    print("=" * 70)
    
    try:
        # Step 1: Check file structure
        if not check_file_structure():
            print("❌ Required files missing. Please check your repository structure.")
            return False
        
        # Step 2: Install dependencies
        install_dependencies()
        
        # Step 3: Install ngrok
        install_ngrok()
        
        # Step 4: Create frontend files FIRST (before starting server)
        print("\n📄 Creating frontend files...")
        create_simple_frontend()
        create_test_interface()
        
        # Step 4.5: Verify frontend files exist and are accessible
        if not verify_frontend_files():
            print("❌ Frontend files verification failed")
            return False
        
        # Step 5: Start TTS server (now with frontend files available)
        if not start_tts_server():
            print("❌ Failed to start TTS server")
            return False
        
        # Step 6: Create ngrok tunnel
        public_url = create_ngrok_tunnel()
        if not public_url:
            print("❌ Failed to create ngrok tunnel")
            return False
        
        # Step 7: Test frontend access
        test_frontend_access(public_url)
        
        # Final output
        print("\n" + "=" * 70)
        print("🎉 DEPLOYMENT COMPLETE!")
        print("=" * 70)
        
        print(f"🌐 Public URL: {public_url}")
        print(f"🔗 Main App: {public_url}")
        print(f"🔗 Frontend: {public_url}/frontend")
        print(f"🔗 Test Interface: {public_url}/test")
        print(f"📚 API Documentation: {public_url}/docs")
        
        print("\n📱 How to use:")
        print("1. Open the public URL in your browser")
        print("2. Your index.html will be served at the root URL")
        print("3. Use the test interface to verify TTS functionality")
        print("4. The server runs in the background on Colab")
        
        print("\n🔧 Troubleshooting:")
        print("• If frontend doesn't load, check server.log for errors")
        print("• If ngrok fails, check ngrok.log for errors")
        print("• Server runs on port 8001, ngrok creates public tunnel")
        print("• Frontend files are served from /static/ and root /")
        
        print("\n📁 Files created:")
        print("• server.log - TTS server logs")
        print("• ngrok.log - Ngrok tunnel logs")
        print("• test_tts.html - Test interface")
        print("• frontend/index.html - Simple frontend for testing")
        
        print("\n🚀 Your TTS app with frontend is now live on Google Colab!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Deployment failed: {e}")
        print("Check the error messages above for troubleshooting steps")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ Deployment completed successfully!")
    else:
        print("\n❌ Deployment failed. Check the logs above.")
