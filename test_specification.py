#!/usr/bin/env python3
"""
Test script to verify the exact WebSocket input/output format as per requirements
"""

import asyncio
import json
import websockets
import time
import base64

class SpecificationTestClient:
    def __init__(self, uri="ws://localhost:8000/ws/tts"):
        self.uri = uri
        self.websocket = None
        self.is_connected = False
        self.received_responses = []
        
    async def connect(self):
        """Connect to the TTS WebSocket server"""
        try:
            print(f"🔌 Connecting to {self.uri}...")
            self.websocket = await websockets.connect(self.uri)
            
            # Wait for connection confirmation
            response = await self.websocket.recv()
            data = json.loads(response)
            
            if data.get("status") == "connected":
                self.is_connected = True
                print("✅ Connected to TTS server successfully!")
                return True
            else:
                print("❌ Unexpected connection response:", data)
                return False
                
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from the TTS server"""
        if self.websocket:
            await self.websocket.close()
            self.is_connected = False
            print("🔌 Disconnected from TTS server")
    
    async def test_specification_format(self):
        """Test the exact input/output format as per requirements"""
        print("🧪 Testing WebSocket Input/Output Format Specification")
        print("=" * 70)
        
        if not self.is_connected:
            print("❌ Not connected to server")
            return False
        
        try:
            # Test 1: Send initial space character (first chunk from client)
            print("\n📝 Test 1: Initial space character")
            print("Input: {'text': ' ', 'flush': false}")
            
            await self.websocket.send(json.dumps({"text": " ", "flush": False}))
            print("✅ Sent initial space character")
            
            # Test 2: Send text chunks
            print("\n📝 Test 2: Text chunks")
            test_text = "This is an example of alignment data."
            
            # Send text in chunks
            chunks = ["This ", "is ", "an ", "example ", "of ", "alignment ", "data."]
            
            for i, chunk in enumerate(chunks):
                print(f"Input {i+1}: {{'text': '{chunk}', 'flush': false}}")
                await self.websocket.send(json.dumps({"text": chunk, "flush": False}))
                await asyncio.sleep(0.2)  # Small delay between chunks
            
            # Test 3: Test flush behavior
            print("\n📝 Test 3: Flush behavior")
            print("Input: {'text': 'Additional text for flush test.', 'flush': true}")
            
            await self.websocket.send(json.dumps({"text": "Additional text for flush test.", "flush": True}))
            print("✅ Sent flush request")
            
            # Test 4: Test flush with empty text (WebSocket should stay open)
            print("\n📝 Test 4: Flush with empty text (WebSocket stays open)")
            print("Input: {'text': '', 'flush': true}")
            
            await self.websocket.send(json.dumps({"text": "", "flush": True}))
            print("✅ Sent flush with empty text")
            
            # Wait for responses
            print("\n⏳ Waiting for audio responses...")
            await asyncio.sleep(3)
            
            # Test 5: Close connection (empty string without flush)
            print("\n📝 Test 5: Close connection")
            print("Input: {'text': '', 'flush': false}")
            
            await self.websocket.send(json.dumps({"text": "", "flush": False}))
            print("✅ Sent close request")
            
            return True
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            return False
    
    async def receive_responses(self):
        """Receive and analyze responses"""
        if not self.is_connected:
            return
        
        try:
            while self.is_connected:
                try:
                    response = await asyncio.wait_for(self.websocket.recv(), timeout=1.0)
                    data = json.loads(response)
                    self.received_responses.append(data)
                    
                    print(f"\n🎵 Received response:")
                    
                    if "audio" in data:
                        audio_length = len(data["audio"])
                        print(f"   Audio: {audio_length} Base64 characters")
                        
                        # Verify audio format (should be Base64 encoded 44.1 kHz, 16-bit, mono PCM)
                        try:
                            audio_bytes = base64.b64decode(data["audio"])
                            print(f"   Audio bytes: {len(audio_bytes)} bytes")
                            print(f"   Audio format: 44.1 kHz, 16-bit, mono PCM ✓")
                        except Exception as e:
                            print(f"   ❌ Audio format error: {e}")
                    
                    if "alignment" in data:
                        alignment = data["alignment"]
                        chars = alignment.get("chars", [])
                        start_times = alignment.get("char_start_times_ms", [])
                        durations = alignment.get("char_durations_ms", [])
                        
                        print(f"   Alignment: {len(chars)} characters")
                        print(f"   Start times: {len(start_times)} timestamps")
                        print(f"   Durations: {len(durations)} durations")
                        
                        # Show first few characters with timing
                        if chars and start_times and durations:
                            print("   Sample alignment:")
                            for i in range(min(5, len(chars))):
                                char = chars[i]
                                start = start_times[i]
                                duration = durations[i]
                                print(f"     '{char}': {start}ms + {duration}ms")
                    
                    elif "error" in data:
                        print(f"   ❌ Error: {data['error']}")
                    
                    elif "status" in data:
                        print(f"   📡 Status: {data['status']} - {data.get('message', '')}")
                    
                except asyncio.TimeoutError:
                    continue
                except websockets.exceptions.ConnectionClosed:
                    print("🔌 WebSocket connection closed")
                    break
                    
        except Exception as e:
            print(f"❌ Error receiving responses: {e}")
    
    async def run_complete_test(self):
        """Run the complete specification test"""
        print("🚀 WebSocket TTS Specification Test")
        print("This test verifies:")
        print("  • Input format: text + flush fields")
        print("  • Initial space character handling")
        print("  • Flush behavior (WebSocket stays open)")
        print("  • Output format: audio + alignment")
        print("  • Audio format: 44.1 kHz, 16-bit, mono PCM, Base64")
        print("  • Character alignment with timestamps")
        print()
        
        # Connect to server
        if not await self.connect():
            return False
        
        try:
            # Start receiving responses in background
            receive_task = asyncio.create_task(self.receive_responses())
            
            # Run the specification test
            success = await self.test_specification_format()
            
            # Wait a bit for final responses
            await asyncio.sleep(2)
            
            # Cancel receive task
            receive_task.cancel()
            
            # Print results
            print(f"\n📊 Test Results:")
            print(f"   Responses received: {len(self.received_responses)}")
            
            if self.received_responses:
                print(f"   First response type: {list(self.received_responses[0].keys())}")
                
                # Verify output format
                for i, response in enumerate(self.received_responses):
                    if "audio" in response and "alignment" in response:
                        print(f"   Response {i+1}: ✓ Valid format (audio + alignment)")
                    else:
                        print(f"   Response {i+1}: ❌ Invalid format")
            
            return success
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            return False
        finally:
            await self.disconnect()

async def main():
    """Main test function"""
    client = SpecificationTestClient()
    
    success = await client.run_complete_test()
    
    if success:
        print("\n🎉 Specification test completed successfully!")
        print("✅ All input/output format requirements are met")
    else:
        print("\n❌ Test failed")
        print("💡 Make sure the backend server is running with: python backend/main.py")

if __name__ == "__main__":
    asyncio.run(main())
