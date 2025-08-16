#!/usr/bin/env python3
"""
Demo script for the SigIQ TTS WebSocket System
Showcases various text examples and streaming capabilities
"""

import asyncio
import json
import websockets
import time
import random

class TTSDemo:
    def __init__(self):
        self.websocket = None
        self.is_connected = False
        
    async def connect(self, uri="ws://localhost:8000/ws/tts"):
        """Connect to the TTS WebSocket server"""
        try:
            print(f"🔌 Connecting to {uri}...")
            self.websocket = await websockets.connect(uri)
            
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
    
    async def send_text(self, text, flush=False):
        """Send text to the TTS server"""
        if not self.is_connected:
            print("❌ Not connected to server")
            return None
        
        message = {"text": text, "flush": flush}
        await self.websocket.send(json.dumps(message))
        print(f"📤 Sent: '{text[:50]}{'...' if len(text) > 50 else ''}' (flush: {flush})")
        
        # Wait for response
        try:
            response = await asyncio.wait_for(self.websocket.recv(), timeout=10.0)
            data = json.loads(response)
            
            if "audio" in data and "alignment" in data:
                print(f"✅ Received audio: {len(data['audio'])} chars, {len(data['alignment']['chars'])} characters aligned")
                return data
            elif "error" in data:
                print(f"❌ Server error: {data['error']}")
                return None
            else:
                print(f"⚠️  Unexpected response: {data}")
                return None
                
        except asyncio.TimeoutError:
            print("⏰ Timeout waiting for response")
            return None
    
    async def demo_basic_tts(self):
        """Demo basic TTS functionality"""
        print("\n🎤 Demo 1: Basic TTS")
        print("=" * 30)
        
        # Send initial space
        await self.send_text(" ", False)
        
        # Send simple text
        text = "Hello, welcome to the SigIQ TTS system. This is a demonstration of text-to-speech capabilities."
        result = await self.send_text(text, False)
        
        if result:
            print("✅ Basic TTS demo completed successfully!")
        else:
            print("❌ Basic TTS demo failed!")
    
    async def demo_streaming(self):
        """Demo streaming text in chunks"""
        print("\n🌊 Demo 2: Streaming Text")
        print("=" * 30)
        
        # Send initial space
        await self.send_text(" ", False)
        
        # Long text to stream
        long_text = """
        This is a demonstration of streaming text-to-speech. 
        The system processes text in chunks and generates audio in real-time. 
        This approach allows for low-latency audio generation and provides 
        character-level timing information for each piece of text.
        """
        
        # Split into chunks
        chunks = [chunk.strip() for chunk in long_text.split('\n') if chunk.strip()]
        
        print(f"📝 Streaming {len(chunks)} text chunks...")
        
        for i, chunk in enumerate(chunks):
            print(f"   Chunk {i+1}/{len(chunks)}: '{chunk[:40]}{'...' if len(chunk) > 40 else ''}'")
            result = await self.send_text(chunk, False)
            
            if not result:
                print(f"❌ Failed to process chunk {i+1}")
                break
            
            # Small delay between chunks
            await asyncio.sleep(0.5)
        
        print("✅ Streaming demo completed!")
    
    async def demo_flush_functionality(self):
        """Demo flush functionality"""
        print("\n⚡ Demo 3: Flush Functionality")
        print("=" * 30)
        
        # Send initial space
        await self.send_text(" ", False)
        
        # Send text without flush
        text1 = "This text will be buffered."
        await self.send_text(text1, False)
        
        # Send more text without flush
        text2 = " This text will also be buffered."
        await self.send_text(text2, False)
        
        # Now flush to generate audio for all buffered text
        print("🔄 Flushing buffer...")
        result = await self.send_text("", True)
        
        if result:
            print("✅ Flush demo completed successfully!")
        else:
            print("❌ Flush demo failed!")
    
    async def demo_various_text_types(self):
        """Demo different types of text content"""
        print("\n📚 Demo 4: Various Text Types")
        print("=" * 30)
        
        # Send initial space
        await self.send_text(" ", False)
        
        # Different text examples
        examples = [
            "Numbers: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10.",
            "Punctuation test: Hello! How are you? This is great...",
            "Special characters: @#$%^&*()_+-=[]{}|;':\",./<>?",
            "Long sentence with many words that should demonstrate the system's ability to handle substantial text input and generate appropriate audio output.",
            "Short words: a, an, the, in, on, at, to, for, of, with, by."
        ]
        
        for i, example in enumerate(examples):
            print(f"📝 Example {i+1}: '{example[:50]}{'...' if len(example) > 50 else ''}'")
            result = await self.send_text(example, False)
            
            if not result:
                print(f"❌ Failed to process example {i+1}")
                break
            
            await asyncio.sleep(0.3)
        
        print("✅ Various text types demo completed!")
    
    async def demo_performance(self):
        """Demo performance with rapid text input"""
        print("\n🚀 Demo 5: Performance Test")
        print("=" * 30)
        
        # Send initial space
        await self.send_text(" ", False)
        
        # Generate many small text chunks rapidly
        start_time = time.time()
        successful_chunks = 0
        total_chunks = 20
        
        for i in range(total_chunks):
            text = f"Chunk {i+1} of {total_chunks} for performance testing."
            result = await self.send_text(text, False)
            
            if result:
                successful_chunks += 1
            
            # Minimal delay
            await asyncio.sleep(0.1)
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"📊 Performance Results:")
        print(f"   Total chunks: {total_chunks}")
        print(f"   Successful: {successful_chunks}")
        print(f"   Failed: {total_chunks - successful_chunks}")
        print(f"   Total time: {duration:.2f} seconds")
        print(f"   Average time per chunk: {duration/total_chunks:.3f} seconds")
        print(f"   Success rate: {(successful_chunks/total_chunks)*100:.1f}%")
        
        if successful_chunks == total_chunks:
            print("✅ Performance demo completed successfully!")
        else:
            print("⚠️  Performance demo completed with some failures")
    
    async def run_all_demos(self):
        """Run all demo functions"""
        print("🎬 SigIQ TTS System Demo Suite")
        print("=" * 50)
        
        if not await self.connect():
            print("❌ Cannot run demos without connection")
            return
        
        try:
            # Run all demos
            await self.demo_basic_tts()
            await asyncio.sleep(1)
            
            await self.demo_streaming()
            await asyncio.sleep(1)
            
            await self.demo_flush_functionality()
            await asyncio.sleep(1)
            
            await self.demo_various_text_types()
            await asyncio.sleep(1)
            
            await self.demo_performance()
            
            print("\n🎉 All demos completed successfully!")
            
        except Exception as e:
            print(f"❌ Demo error: {e}")
        
        finally:
            await self.disconnect()

async def main():
    """Main function"""
    demo = TTSDemo()
    await demo.run_all_demos()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        print("💡 Make sure the backend server is running: cd backend && python main.py")
