#!/usr/bin/env python3
"""
Test script to verify the audio format being sent by the backend
"""

import asyncio
import json
import websockets
import base64
import struct

async def test_audio_format():
    """Test the audio format from the backend"""
    uri = "ws://localhost:8001/ws/tts"
    
    try:
        print("🔌 Connecting to WebSocket server...")
        async with websockets.connect(uri) as websocket:
            print("✅ Connected successfully")
            
            # Wait for connection confirmation
            response = await websocket.recv()
            data = json.loads(response)
            print(f"📡 Connection response: {data}")
            
            # Send initial space character
            print("\n📝 Sending initial space character...")
            await websocket.send(json.dumps({"text": " ", "flush": False}))
            
            # Send test text
            test_text = "Hello world"
            print(f"\n📝 Sending test text: '{test_text}'")
            await websocket.send(json.dumps({"text": test_text, "flush": True}))
            
            # Wait for audio response
            print("\n⏳ Waiting for audio response...")
            response = await websocket.recv()
            data = json.loads(response)
            
            if "audio" in data and "alignment" in data:
                print("✅ Audio response received!")
                
                # Analyze audio data
                audio_base64 = data["audio"]
                audio_bytes = base64.b64decode(audio_base64)
                
                print(f"\n📊 Audio Analysis:")
                print(f"   Base64 length: {len(audio_base64)} characters")
                print(f"   Decoded bytes: {len(audio_bytes)} bytes")
                print(f"   Expected samples: {len(audio_bytes) // 2} (16-bit)")
                print(f"   Expected duration: {(len(audio_bytes) // 2) / 44100:.3f}s")
                
                # Check if bytes are even (required for 16-bit samples)
                if len(audio_bytes) % 2 != 0:
                    print("   ❌ WARNING: Odd number of bytes (not valid 16-bit PCM)")
                else:
                    print("   ✅ Valid 16-bit PCM byte count")
                
                # Analyze first few samples
                print(f"\n🔍 First 10 PCM samples:")
                for i in range(min(10, len(audio_bytes) // 2)):
                    sample_index = i * 2
                    if sample_index + 1 < len(audio_bytes):
                        # Try both endianness
                        sample_le = struct.unpack('<h', audio_bytes[sample_index:sample_index + 2])[0]  # Little-endian
                        sample_be = struct.unpack('>h', audio_bytes[sample_index:sample_index + 2])[0]  # Big-endian
                        
                        print(f"   Sample {i}: LE={sample_le:6d}, BE={sample_be:6d}, Raw=[{audio_bytes[sample_index]:3d}, {audio_bytes[sample_index + 1]:3d}]")
                
                # Check alignment data
                alignment = data["alignment"]
                print(f"\n📺 Alignment Data:")
                print(f"   Characters: {len(alignment['chars'])}")
                print(f"   Start times: {len(alignment['char_start_times_ms'])}")
                print(f"   Durations: {len(alignment['char_durations_ms'])}")
                
                if alignment['chars']:
                    print(f"   Sample: '{''.join(alignment['chars'][:10])}'")
                
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
    
    print("\n🎉 Audio format test completed!")
    return True

if __name__ == "__main__":
    asyncio.run(test_audio_format())
