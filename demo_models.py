#!/usr/bin/env python3
"""
Demo script showing how to use both Kokoro and Chatterbox TTS models
"""

import requests
import json
import time
import base64
import io
import wave
import numpy as np

# Server configuration
BASE_URL = "http://localhost:8001"

def switch_to_model(model_type):
    """Switch to the specified TTS model"""
    print(f"🔄 Switching to {model_type} model...")
    try:
        response = requests.post(f"{BASE_URL}/models/switch/{model_type}")
        if response.status_code == 200:
            result = response.json()
            print(f"✅ {result['message']}")
            return True
        else:
            print(f"❌ Failed to switch: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def get_current_model():
    """Get current model information"""
    try:
        response = requests.get(f"{BASE_URL}/models/current")
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except:
        return None

def test_text_to_speech(text, model_type):
    """Test text-to-speech with the specified model"""
    print(f"\n🎤 Testing {model_type.upper()} TTS with text: '{text[:50]}...'")
    
    # Switch to the model
    if not switch_to_model(model_type):
        print(f"❌ Failed to switch to {model_type}")
        return False
    
    # Wait for model initialization
    time.sleep(2)
    
    # Verify current model
    current = get_current_model()
    if not current or current['model_type'] != model_type:
        print(f"❌ Model switch verification failed")
        return False
    
    print(f"✅ Using {model_type} model")
    
    # Here you would typically test the WebSocket TTS functionality
    # For this demo, we'll just show the model info
    print(f"   Model type: {current['model_type']}")
    print(f"   Device: {current['device']}")
    print(f"   Configuration: {current['current_config']}")
    
    return True

def main():
    """Main demo function"""
    print("🎭 TTS Model Demo - Kokoro vs Chatterbox")
    print("=" * 50)
    
    # Test text
    test_text = "Hello! This is a test of the text-to-speech system. We can switch between different models seamlessly."
    
    # Test with Chatterbox
    print("\n🔵 Testing Chatterbox Model")
    print("-" * 30)
    test_text_to_speech(test_text, "chatterbox")
    
    # Test with Kokoro
    print("\n🟢 Testing Kokoro Model")
    print("-" * 30)
    test_text_to_speech(test_text, "kokoro")
    
    # Switch back to Chatterbox
    print("\n🔄 Switching back to Chatterbox...")
    switch_to_model("chatterbox")
    
    print("\n✅ Demo completed!")
    print("\nTo test the actual TTS functionality:")
    print("1. Start the server: python backend/main.py")
    print("2. Use the WebSocket endpoint: ws://localhost:8001/ws/tts")
    print("3. Switch models using: POST /models/switch/{model_type}")
    print("4. Check model info: GET /models/current")

if __name__ == "__main__":
    main()
