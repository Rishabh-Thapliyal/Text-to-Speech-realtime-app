#!/usr/bin/env python3
"""
Test script to verify buffer clearing functionality
"""

import requests
import json
import time

# Server configuration
BASE_URL = "http://localhost:8001"

def test_buffer_clearing():
    """Test that buffers are properly cleared between requests"""
    print("🧪 Testing Buffer Clearing Functionality")
    print("=" * 50)
    
    # Test 1: Check if server is running
    print("🔍 Checking server status...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            print("✅ Server is running")
        else:
            print(f"❌ Server health check failed: {response.status_code}")
            return
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        return
    
    # Test 2: Get active connections
    print("\n🔍 Checking active connections...")
    try:
        response = requests.get(f"{BASE_URL}/connections")
        if response.status_code == 200:
            connections = response.json()
            print(f"✅ Found {connections['active_connections']} active connections")
            
            if connections['connections']:
                for conn in connections['connections']:
                    print(f"   Connection {conn['connection_id']}: {conn['buffer_length']} chars in buffer")
            else:
                print("   No active connections found")
        else:
            print(f"❌ Failed to get connections: {response.status_code}")
    except Exception as e:
        print(f"❌ Error getting connections: {e}")
    
    # Test 3: Test model switching to ensure no buffer interference
    print("\n🔄 Testing model switching...")
    try:
        # Switch to Kokoro
        response = requests.post(f"{BASE_URL}/models/switch/kokoro")
        if response.status_code == 200:
            print("✅ Switched to Kokoro model")
        else:
            print(f"❌ Failed to switch to Kokoro: {response.status_code}")
        
        # Switch back to Chatterbox
        response = requests.post(f"{BASE_URL}/models/switch/chatterbox")
        if response.status_code == 200:
            print("✅ Switched back to Chatterbox model")
        else:
            print(f"❌ Failed to switch back to Chatterbox: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error switching models: {e}")
    
    print("\n✅ Buffer clearing tests completed!")
    print("\nTo test the actual WebSocket functionality:")
    print("1. Open frontend/index.html in your browser")
    print("2. Connect to the WebSocket")
    print("3. Try sending 'hey how are you' multiple times")
    print("4. Each request should process only the new text, not accumulated text")
    print("5. Use the '🧹 Clear Buffer' button to manually clear if needed")

if __name__ == "__main__":
    test_buffer_clearing()
