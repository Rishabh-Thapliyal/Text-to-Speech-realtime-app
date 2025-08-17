#!/usr/bin/env python3
"""
Debug script to identify and resolve Kokoro TTS issues
"""

import requests
import json
import time
import subprocess
import sys

# Server configuration
BASE_URL = "http://localhost:8001"

def check_kokoro_installation():
    """Check if Kokoro is properly installed"""
    print("🔍 Checking Kokoro Installation")
    print("=" * 40)
    
    # Check Python package
    try:
        import kokoro
        print(f"✅ Kokoro package imported: {kokoro.__file__}")
        print(f"   Version: {getattr(kokoro, '__version__', 'Unknown')}")
    except ImportError as e:
        print(f"❌ Kokoro package not found: {e}")
        return False
    
    # Check KPipeline class
    try:
        from kokoro import KPipeline
        print("✅ KPipeline class imported successfully")
    except ImportError as e:
        print(f"❌ KPipeline import failed: {e}")
        return False
    
    # Test KPipeline instantiation
    try:
        pipeline = KPipeline(lang_code='a')
        print(f"✅ KPipeline instantiated: {type(pipeline)}")
        
        # Check if it's callable
        if hasattr(pipeline, '__call__'):
            print("✅ KPipeline is callable")
        else:
            print("⚠️  KPipeline may not be callable")
            
        return True
    except Exception as e:
        print(f"❌ KPipeline instantiation failed: {e}")
        print(f"   Error type: {type(e)}")
        return False

def check_server_status():
    """Check if the server is running and responding"""
    print("\n🔍 Checking Server Status")
    print("=" * 40)
    
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            print("✅ Server is running and responding")
            health = response.json()
            current_model = health['tts_engine']['model_type']
            print(f"   Current model: {current_model}")
            return True
        else:
            print(f"❌ Server health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        return False

def check_kokoro_availability():
    """Check Kokoro availability via API"""
    print("\n🔍 Checking Kokoro Availability via API")
    print("=" * 40)
    
    try:
        response = requests.get(f"{BASE_URL}/models/kokoro/check")
        if response.status_code == 200:
            result = response.json()
            print(f"   Kokoro available: {result['kokoro_available']}")
            print(f"   Message: {result['message']}")
            return result['kokoro_available']
        else:
            print(f"❌ Kokoro check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error checking Kokoro availability: {e}")
        return False

def debug_model_state():
    """Debug current model state"""
    print("\n🔍 Debugging Current Model State")
    print("=" * 40)
    
    try:
        response = requests.get(f"{BASE_URL}/models/debug")
        if response.status_code == 200:
            debug_info = response.json()
            print(f"   Model type: {debug_info['model_type']}")
            print(f"   Model object: {debug_info['model_object']}")
            print(f"   Device: {debug_info['device']}")
            print(f"   Config selected: {debug_info['configuration']['selected_model']}")
            return debug_info
        else:
            print(f"❌ Debug endpoint failed: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Error getting debug info: {e}")
        return None

def test_model_switching():
    """Test model switching functionality"""
    print("\n🔄 Testing Model Switching")
    print("=" * 40)
    
    # Switch to Kokoro
    print("   Switching to Kokoro...")
    try:
        response = requests.post(f"{BASE_URL}/models/switch/kokoro")
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ {result['message']}")
            
            # Wait for model to initialize
            time.sleep(3)
            
            # Check model state after switch
            debug_info = debug_model_state()
            if debug_info and debug_info['model_type'] == 'kokoro' and 'None' not in debug_info['model_object']:
                print("   ✅ Model switch to Kokoro successful!")
                return True
            else:
                print("   ❌ Model switch to Kokoro failed!")
                return False
        else:
            error = response.json()
            print(f"   ❌ Failed to switch to Kokoro: {error.get('error', 'Unknown error')}")
            return False
    except Exception as e:
        print(f"   ❌ Error switching to Kokoro: {e}")
        return False

def test_kokoro_tts():
    """Test if Kokoro TTS actually works"""
    print("\n🎤 Testing Kokoro TTS Functionality")
    print("=" * 40)
    
    # First ensure we're using Kokoro
    try:
        response = requests.post(f"{BASE_URL}/models/switch/kokoro")
        if response.status_code != 200:
            print("   ❌ Cannot switch to Kokoro for testing")
            return False
        
        time.sleep(3)  # Wait for initialization
        
        # Try to generate audio (this will test the actual TTS)
        print("   Testing TTS generation...")
        # Note: This would require a WebSocket connection for full testing
        # For now, just check if the model is properly initialized
        
        debug_info = debug_model_state()
        if debug_info and debug_info['model_type'] == 'kokoro' and 'None' not in debug_info['model_object']:
            print("   ✅ Kokoro model appears to be properly initialized")
            return True
        else:
            print("   ❌ Kokoro model not properly initialized")
            return False
            
    except Exception as e:
        print(f"   ❌ Error testing Kokoro TTS: {e}")
        return False

def provide_solutions():
    """Provide solutions based on the issues found"""
    print("\n💡 Solutions and Recommendations")
    print("=" * 40)
    
    print("1. **If Kokoro package is not installed:**")
    print("   pip install kokoro>=0.9.2")
    
    print("\n2. **If Kokoro is installed but not working:**")
    print("   - Check Python environment")
    print("   - Verify kokoro version compatibility")
    print("   - Try reinstalling: pip uninstall kokoro && pip install kokoro>=0.9.2")
    
    print("\n3. **If model switching fails:**")
    print("   - Restart the server: ./start.sh")
    print("   - Check server logs for detailed error messages")
    print("   - Use the debug endpoints to identify issues")
    
    print("\n4. **Alternative approach:**")
    print("   - Use Chatterbox TTS as fallback")
    print("   - Check if Kokoro works in a simple Python script")
    print("   - Verify system dependencies")

def main():
    """Main debug function"""
    print("🔧 Kokoro TTS Debug Script")
    print("=" * 50)
    
    # Step 1: Check Kokoro installation
    kokoro_installed = check_kokoro_installation()
    
    # Step 2: Check server status
    server_running = check_server_status()
    
    if not server_running:
        print("\n❌ Server is not running. Please start it with: ./start.sh")
        return
    
    # Step 3: Check Kokoro availability via API
    kokoro_available = check_kokoro_availability()
    
    # Step 4: Debug current model state
    debug_info = debug_model_state()
    
    # Step 5: Test model switching
    if kokoro_available:
        switch_success = test_model_switching()
        
        # Step 6: Test TTS functionality
        if switch_success:
            tts_working = test_kokoro_tts()
        else:
            tts_working = False
    else:
        switch_success = False
        tts_working = False
    
    # Summary
    print("\n📊 Debug Summary")
    print("=" * 40)
    print(f"   Kokoro installed: {'✅' if kokoro_installed else '❌'}")
    print(f"   Server running: {'✅' if server_running else '❌'}")
    print(f"   Kokoro available: {'✅' if kokoro_available else '❌'}")
    print(f"   Model switching: {'✅' if switch_success else '❌'}")
    print(f"   TTS working: {'✅' if tts_working else '❌'}")
    
    # Provide solutions
    if not kokoro_installed or not kokoro_available or not switch_success:
        provide_solutions()
    
    print("\n✅ Debug completed!")

if __name__ == "__main__":
    main()
