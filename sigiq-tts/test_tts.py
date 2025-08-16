#!/usr/bin/env python3
"""
Simple test script for the SigIQ TTS WebSocket System
Tests the TTS functionality directly without WebSocket overhead
"""

import asyncio
import json
import websockets
import time
import base64
import wave
import tempfile
import os

async def test_tts_websocket():
    """Test the TTS WebSocket functionality"""
    print("🧪 Testing SigIQ TTS WebSocket System")
    print("=" * 50)
    
    # WebSocket URL
    uri = "ws://localhost:8000/ws/tts"
    
    try:
        print(f"🔌 Connecting to {uri}...")
        async with websockets.connect(uri) as websocket:
            print("✅ WebSocket connected successfully")
            
            # Wait for connection confirmation
            response = await websocket.recv()
            data = json.loads(response)
            print(f"📡 Connection response: {data}")
            
            # Test 1: Send initial space character
            print("\n📝 Test 1: Sending initial space character")
            await websocket.send(json.dumps({"text": " ", "flush": False}))
            
            # Test 2: Send some text
            test_text = "Hello, this is a test of the TTS system."
            print(f"\n📝 Test 2: Sending text: '{test_text}'")
            await websocket.send(json.dumps({"text": test_text, "flush": False}))
            
            # Wait for audio response
            print("⏳ Waiting for audio response...")
            response = await websocket.recv()
            data = json.loads(response)
            
            if "audio" in data and "alignment" in data:
                print("✅ Audio response received successfully!")
                print(f"📊 Audio data length: {len(data['audio'])} characters (Base64)")
                print(f"📊 Alignment data: {len(data['alignment']['chars'])} characters")
                
                # Display alignment info
                alignment = data['alignment']
                print(f"📺 Character alignment:")
                for i, char in enumerate(alignment['chars'][:10]):  # Show first 10 chars
                    start_time = alignment['char_start_times_ms'][i]
                    duration = alignment['char_durations_ms'][i]
                    print(f"   '{char}': {start_time}ms + {duration}ms")
                
                if len(alignment['chars']) > 10:
                    print(f"   ... and {len(alignment['chars']) - 10} more characters")
                
                # Test 3: Test flush functionality
                print("\n📝 Test 3: Testing flush functionality")
                await websocket.send(json.dumps({"text": "Additional text for flush test.", "flush": True}))
                
                # Wait for flushed audio response
                print("⏳ Waiting for flushed audio response...")
                response = await websocket.recv()
                data = json.loads(response)
                
                if "audio" in data:
                    print("✅ Flushed audio response received!")
                    print(f"📊 Flushed audio length: {len(data['audio'])} characters")
                
                # Test 4: Close connection
                print("\n📝 Test 4: Closing connection")
                await websocket.send(json.dumps({"text": "", "flush": False}))
                
            else:
                print("❌ Unexpected response format:")
                print(json.dumps(data, indent=2))
                
    except websockets.exceptions.ConnectionRefused:
        print("❌ Connection refused. Is the backend server running?")
        print("💡 Start the backend with: cd backend && python main.py")
        return False
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        return False
    
    print("\n🎉 All tests completed successfully!")
    return True

def test_tts_direct():
    """Test TTS functionality directly without WebSocket"""
    print("\n🧪 Testing TTS Engine Directly")
    print("=" * 40)
    
    try:
        import pyttsx3
        
        # Initialize TTS engine
        engine = pyttsx3.init()
        
        # Test basic TTS
        test_text = "This is a direct TTS test."
        print(f"📝 Testing text: '{test_text}'")
        
        # Get available voices
        voices = engine.getProperty('voices')
        print(f"🎤 Available voices: {len(voices)}")
        for i, voice in enumerate(voices[:3]):  # Show first 3 voices
            print(f"   Voice {i}: {voice.name} ({voice.id})")
        
        # Set properties
        engine.setProperty('rate', 150)
        engine.setProperty('volume', 0.9)
        
        print("🔊 Playing audio... (you should hear speech)")
        engine.say(test_text)
        engine.runAndWait()
        
        print("✅ Direct TTS test completed successfully!")
        return True
        
    except ImportError:
        print("❌ pyttsx3 not available. Install with: pip install pyttsx3")
        return False
    except Exception as e:
        print(f"❌ Direct TTS test failed: {e}")
        return False

def test_audio_processing():
    """Test audio processing capabilities"""
    print("\n🧪 Testing Audio Processing")
    print("=" * 30)
    
    try:
        import numpy as np
        
        # Create a simple test audio signal
        sample_rate = 44100
        duration = 1.0  # 1 second
        frequency = 440  # A4 note
        
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        audio_data = np.sin(2 * np.pi * frequency * t) * 0.3
        
        # Convert to 16-bit PCM
        audio_16bit = (audio_data * 32767).astype(np.int16)
        
        # Test Base64 encoding/decoding
        audio_bytes = audio_16bit.tobytes()
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        decoded_bytes = base64.b64decode(audio_base64)
        
        print(f"📊 Test audio: {sample_rate}Hz, {duration}s, {frequency}Hz")
        print(f"📊 Original size: {len(audio_bytes)} bytes")
        print(f"📊 Base64 size: {len(audio_base64)} characters")
        print(f"📊 Decoded size: {len(decoded_bytes)} bytes")
        
        if audio_bytes == decoded_bytes:
            print("✅ Base64 encoding/decoding test passed!")
        else:
            print("❌ Base64 encoding/decoding test failed!")
            return False
        
        return True
        
    except ImportError:
        print("❌ numpy not available. Install with: pip install numpy")
        return False
    except Exception as e:
        print(f"❌ Audio processing test failed: {e}")
        return False

async def main():
    """Main test function"""
    print("🚀 SigIQ TTS System Test Suite")
    print("=" * 50)
    
    # Test 1: Direct TTS functionality
    tts_success = test_tts_direct()
    
    # Test 2: Audio processing
    audio_success = test_audio_processing()
    
    # Test 3: WebSocket functionality (only if backend is running)
    websocket_success = await test_tts_websocket()
    
    # Summary
    print("\n📊 Test Results Summary")
    print("=" * 30)
    print(f"🎤 Direct TTS: {'✅ PASS' if tts_success else '❌ FAIL'}")
    print(f"🔊 Audio Processing: {'✅ PASS' if audio_success else '❌ FAIL'}")
    print(f"🔌 WebSocket TTS: {'✅ PASS' if websocket_success else '❌ FAIL'}")
    
    if tts_success and audio_success:
        print("\n🎉 Core TTS functionality is working!")
        if websocket_success:
            print("🎉 WebSocket system is fully operational!")
        else:
            print("⚠️  WebSocket test failed - check if backend server is running")
    else:
        print("\n❌ Some core functionality tests failed")
        print("💡 Check the error messages above and install missing dependencies")

if __name__ == "__main__":
    asyncio.run(main())
