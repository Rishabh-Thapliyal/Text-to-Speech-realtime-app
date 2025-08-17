#!/usr/bin/env python3
"""
Test script for model switching between Kokoro and Chatterbox TTS models
"""

import requests
import json
import time

# Server configuration
BASE_URL = "http://localhost:8001"

def test_model_info():
    """Test getting model information"""
    print("🔍 Getting current model information...")
    try:
        response = requests.get(f"{BASE_URL}/models/current")
        if response.status_code == 200:
            model_info = response.json()
            print(f"✅ Current model: {model_info['model_type']}")
            print(f"   Device: {model_info['device']}")
            print(f"   Available models: {model_info['available_models']}")
            return model_info
        else:
            print(f"❌ Failed to get model info: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Error getting model info: {e}")
        return None

def test_available_models():
    """Test getting available models"""
    print("\n🔍 Getting available models...")
    try:
        response = requests.get(f"{BASE_URL}/models")
        if response.status_code == 200:
            models_info = response.json()
            print(f"✅ Available models: {models_info['available_models']}")
            print(f"   Current model: {models_info['current_model']['model_type']}")
            return models_info
        else:
            print(f"❌ Failed to get models: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Error getting models: {e}")
        return None

def test_switch_model(model_type):
    """Test switching to a different model"""
    print(f"\n🔄 Switching to {model_type} model...")
    try:
        response = requests.post(f"{BASE_URL}/models/switch/{model_type}")
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Successfully switched to {model_type}: {result['message']}")
            return True
        else:
            print(f"❌ Failed to switch to {model_type}: {response.status_code}")
            try:
                error = response.json()
                print(f"   Error: {error.get('error', 'Unknown error')}")
            except:
                print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error switching to {model_type}: {e}")
        return False

def test_health_check():
    """Test health check endpoint"""
    print("\n🔍 Testing health check...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            health = response.json()
            print(f"✅ Server status: {health['status']}")
            print(f"   TTS engine: {health['tts_engine']['model_type']}")
            print(f"   Active connections: {health['active_connections']}")
            return health
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Error in health check: {e}")
        return None

def main():
    """Main test function"""
    print("🧪 Testing TTS Model Switching Functionality")
    print("=" * 50)
    
    # Test 1: Get current model info
    current_model = test_model_info()
    if not current_model:
        print("❌ Cannot proceed without model info")
        return
    
    # Test 2: Get available models
    models_info = test_available_models()
    if not models_info:
        print("❌ Cannot proceed without models info")
        return
    
    # Test 3: Health check
    health = test_health_check()
    if not health:
        print("❌ Health check failed")
        return
    
    # Test 4: Switch to different model
    current_model_type = current_model['model_type']
    target_model_type = "kokoro" if current_model_type == "chatterbox" else "chatterbox"
    
    print(f"\n🔄 Testing model switching from {current_model_type} to {target_model_type}...")
    
    # Switch to target model
    if test_switch_model(target_model_type):
        print("⏳ Waiting for model to initialize...")
        time.sleep(2)  # Give some time for model initialization
        
        # Verify the switch
        new_model_info = test_model_info()
        if new_model_info and new_model_info['model_type'] == target_model_type:
            print(f"✅ Model switch verified: now using {target_model_type}")
        else:
            print(f"❌ Model switch verification failed")
    
    # Test 5: Switch back to original model
    print(f"\n🔄 Switching back to original model {current_model_type}...")
    if test_switch_model(current_model_type):
        print("⏳ Waiting for model to initialize...")
        time.sleep(2)
        
        # Verify the switch back
        final_model_info = test_model_info()
        if final_model_info and final_model_info['model_type'] == current_model_type:
            print(f"✅ Model switch back verified: now using {current_model_type}")
        else:
            print(f"❌ Model switch back verification failed")
    
    # Final health check
    print("\n🔍 Final health check...")
    final_health = test_health_check()
    if final_health:
        print("✅ All tests completed successfully!")
    else:
        print("❌ Final health check failed")

if __name__ == "__main__":
    main()
