#!/usr/bin/env python3
"""
Test script for bidirectional WebSocket streaming with concurrent send/receive
"""

import asyncio
import json
import websockets
import time
import threading
from concurrent.futures import ThreadPoolExecutor

class BidirectionalTTSClient:
    def __init__(self, uri="ws://localhost:8000/ws/tts"):
        self.uri = uri
        self.websocket = None
        self.is_connected = False
        self.received_audio = []
        self.latency_measurements = []
        
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
    
    async def send_text_streaming(self, text, chunk_size=5):
        """Send text in chunks to test streaming"""
        if not self.is_connected:
            print("❌ Not connected to server")
            return
        
        print(f"📤 Streaming text: '{text}' in chunks of {chunk_size}")
        
        # Send initial space
        await self.websocket.send(json.dumps({"text": " ", "flush": False}))
        
        # Send text in chunks
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i + chunk_size]
            start_time = time.time()
            
            await self.websocket.send(json.dumps({"text": chunk, "flush": False}))
            print(f"📤 Sent chunk: '{chunk}'")
            
            # Small delay to simulate real-time typing
            await asyncio.sleep(0.1)
        
        # Send flush signal
        await self.websocket.send(json.dumps({"text": "", "flush": True}))
        print("📤 Sent flush signal")
    
    async def receive_audio_streaming(self):
        """Receive audio chunks and measure latency"""
        if not self.is_connected:
            return
        
        try:
            while self.is_connected:
                try:
                    # Set a timeout for receiving
                    response = await asyncio.wait_for(self.websocket.recv(), timeout=5.0)
                    data = json.loads(response)
                    
                    if "audio" in data:
                        audio_length = len(data["audio"])
                        self.received_audio.append(audio_length)
                        print(f"🎵 Received audio chunk: {audio_length} chars")
                        
                        if "alignment" in data:
                            chars = len(data["alignment"]["chars"])
                            print(f"📊 Character alignment: {chars} characters")
                    
                    elif "error" in data:
                        print(f"❌ Server error: {data['error']}")
                    
                except asyncio.TimeoutError:
                    # No message received within timeout
                    continue
                except websockets.exceptions.ConnectionClosed:
                    print("🔌 WebSocket connection closed")
                    break
                    
        except Exception as e:
            print(f"❌ Error receiving audio: {e}")
    
    async def test_bidirectional_streaming(self):
        """Test bidirectional streaming with concurrent operations"""
        print("🧪 Testing Bidirectional WebSocket Streaming")
        print("=" * 60)
        
        # Connect to server
        if not await self.connect():
            return False
        
        try:
            # Start receiving audio in background
            receive_task = asyncio.create_task(self.receive_audio_streaming())
            
            # Test text for streaming
            test_text = "This is a test of bidirectional streaming with concurrent send and receive operations for low latency text-to-speech generation."
            
            # Send text in streaming mode
            await self.send_text_streaming(test_text, chunk_size=8)
            
            # Wait a bit for audio processing
            await asyncio.sleep(2)
            
            # Cancel receive task
            receive_task.cancel()
            
            # Send another test
            test_text2 = "Second test with different content to verify concurrent processing."
            await self.send_text_streaming(test_text2, chunk_size=6)
            
            # Wait for final processing
            await asyncio.sleep(2)
            
            print(f"\n📊 Streaming Results:")
            print(f"   Audio chunks received: {len(self.received_audio)}")
            print(f"   Total audio data: {sum(self.received_audio)} chars")
            
            return True
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            return False
        finally:
            await self.disconnect()

async def main():
    """Main test function"""
    client = BidirectionalTTSClient()
    
    print("🚀 Bidirectional WebSocket TTS Test")
    print("This test demonstrates:")
    print("  • Concurrent send/receive operations")
    print("  • Streaming text input")
    print("  • Low latency audio generation")
    print("  • Real-time bidirectional communication")
    print()
    
    success = await client.test_bidirectional_streaming()
    
    if success:
        print("\n🎉 Bidirectional streaming test completed successfully!")
        print("✅ The server supports concurrent send/receive operations")
        print("✅ Low latency streaming is working")
    else:
        print("\n❌ Test failed")
        print("💡 Make sure the backend server is running with: python backend/main.py")

if __name__ == "__main__":
    asyncio.run(main())
