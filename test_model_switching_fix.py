#!/usr/bin/env python3
"""
Test script to verify the model switching fix
"""

import requests
import json
import time

# Server configuration
BASE_URL = "http://localhost:8001"

def test_model_switching_fix():
    """Test that model switching now works correctly"""
    print("🔧 Testing Model Switching Fix")
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
    
    # Test 2: Debug current model state
    print("\n🔍 Debugging current model state...")
    try:
        response = requests.get(f"{BASE_URL}/models/debug")
        if response.status_code == 200:
            debug_info = response.json()
            print(f"   Model type: {debug_info['model_type']}")
            print(f"   Model object: {debug_info['model_object']}")
            print(f"   Device: {debug_info['device']}")
            print(f"   Config selected: {debug_info['configuration']['selected_model']}")
        else:
            print(f"   ❌ Debug endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error getting debug info: {e}")
    
    # Test 3: Test model switching to Kokoro
    print("\n🔄 Testing switch to Kokoro...")
    try:
        response = requests.post(f"{BASE_URL}/models/switch/kokoro")
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ {result['message']}")
            
            # Wait for model to initialize
            time.sleep(3)
            
            # Check model state after switch
            debug_response = requests.get(f"{BASE_URL}/models/debug")
            if debug_response.status_code == 200:
                debug_info = debug_response.json()
                print(f"   After switch - Model type: {debug_info['model_type']}")
                print(f"   After switch - Model object: {debug_info['model_object']}")
                
                if debug_info['model_type'] == 'kokoro':
                    print("   ✅ Model switch to Kokoro successful!")
                else:
                    print(f"   ❌ Model switch failed. Expected: kokoro, Got: {debug_info['model_type']}")
            else:
                print(f"   ❌ Debug check failed: {debug_response.status_code}")
        else:
            error = response.json()
            print(f"   ❌ Failed to switch to Kokoro: {error.get('error', 'Unknown error')}")
    except Exception as e:
        print(f"   ❌ Error switching to Kokoro: {e}")
    
    # Test 4: Test model switching back to Chatterbox
    print("\n🔄 Testing switch back to Chatterbox...")
    try:
        response = requests.post(f"{BASE_URL}/models/switch/chatterbox")
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ {result['message']}")
            
            # Wait for model to initialize
            time.sleep(3)
            
            # Check model state after switch
            debug_response = requests.get(f"{BASE_URL}/models/debug")
            if debug_response.status_code == 200:
                debug_info = debug_response.json()
                print(f"   After switch - Model type: {debug_info['model_type']}")
                print(f"   After switch - Model object: {debug_info['model_object']}")
                
                if debug_info['model_type'] == 'chatterbox':
                    print("   ✅ Model switch to Chatterbox successful!")
                else:
                    print(f"   ❌ Model switch failed. Expected: chatterbox, Got: {debug_info['model_type']}")
            else:
                print(f"   ❌ Debug check failed: {debug_response.status_code}")
        else:
            error = response.json()
            print(f"   ❌ Failed to switch to Chatterbox: {error.get('error', 'Unknown error')}")
    except Exception as e:
        print(f"   ❌ Error switching to Chatterbox: {e}")
    
    # Test 5: Test configuration refresh
    print("\n🔄 Testing configuration refresh...")
    try:
        response = requests.post(f"{BASE_URL}/models/refresh")
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ {result['message']}")
        else:
            error = response.json()
            print(f"   ❌ Configuration refresh failed: {error.get('error', 'Unknown error')}")
    except Exception as e:
        print(f"   ❌ Error refreshing configuration: {e}")
    
    print("\n✅ Model switching fix tests completed!")
    print("\n🎯 Expected Results:")
    print("   • Model switching should work without errors")
    print("   • Model objects should be properly initialized")
    print("   • No more 'ChatterboxTTS object is not callable' errors")
    print("   • Clean model reinitialization on each switch")

if __name__ == "__main__":
    test_model_switching_fix()
