#!/usr/bin/env python3
"""
Test script to verify the model selection UI functionality
"""

import requests
import json
import time

# Server configuration
BASE_URL = "http://localhost:8001"

def test_model_selection_ui():
    """Test the model selection UI functionality"""
    print("🎭 Testing Model Selection UI Functionality")
    print("=" * 50)
    
    # Test 1: Check if server is running
    print("🔍 Checking server status...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            print("✅ Server is running")
            health = response.json()
            current_model = health['tts_engine']['model_type']
            print(f"   Current model: {current_model}")
        else:
            print(f"❌ Server health check failed: {response.status_code}")
            return
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        return
    
    # Test 2: Test model switching API
    print("\n🔄 Testing model switching API...")
    
    # Switch to Kokoro
    print("   Switching to Kokoro...")
    try:
        response = requests.post(f"{BASE_URL}/models/switch/kokoro")
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ {result['message']}")
        else:
            print(f"   ❌ Failed to switch to Kokoro: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error switching to Kokoro: {e}")
    
    # Wait a moment for model to initialize
    time.sleep(2)
    
    # Switch to Chatterbox
    print("   Switching to Chatterbox...")
    try:
        response = requests.post(f"{BASE_URL}/models/switch/chatterbox")
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ {result['message']}")
        else:
            print(f"   ❌ Failed to switch to Chatterbox: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error switching to Chatterbox: {e}")
    
    # Test 3: Test model info endpoints
    print("\n📋 Testing model info endpoints...")
    
    # Get current model
    try:
        response = requests.get(f"{BASE_URL}/models/current")
        if response.status_code == 200:
            model_info = response.json()
            print(f"   ✅ Current model: {model_info['model_type']}")
            print(f"      Device: {model_info['device']}")
        else:
            print(f"   ❌ Failed to get current model: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error getting current model: {e}")
    
    # Get all models
    try:
        response = requests.get(f"{BASE_URL}/models")
        if response.status_code == 200:
            models_info = response.json()
            print(f"   ✅ Available models: {models_info['available_models']}")
        else:
            print(f"   ❌ Failed to get models: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error getting models: {e}")
    
    print("\n✅ Model selection UI tests completed!")
    print("\n🎨 UI Features Available:")
    print("   • Model selection dropdown (Chatterbox/Kokoro)")
    print("   • Switch Model button (when connected)")
    print("   • Refresh Status button")
    print("   • Visual model status indicator")
    print("   • Real-time model switching")
    print("\n🚀 To test the UI:")
    print("1. Open frontend/index.html in your browser")
    print("2. Connect to the WebSocket")
    print("3. Use the model selection dropdown to choose a model")
    print("4. Click '🔄 Switch Model' to switch")
    print("5. Watch the model status indicator change")
    print("6. Use '🔍 Refresh Status' to check current model")

if __name__ == "__main__":
    test_model_selection_ui()
