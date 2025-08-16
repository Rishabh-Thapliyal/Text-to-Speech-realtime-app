#!/usr/bin/env python3
"""
Test script for Chatterbox TTS integration
"""

import torchaudio as ta
from chatterbox.tts import ChatterboxTTS

def test_chatterbox_tts():
    """Test the Chatterbox TTS model"""
    print("🎤 Testing Chatterbox TTS Integration")
    print("=" * 50)
    
    try:
        # Initialize model
        print("🔧 Initializing Chatterbox TTS model...")
        model = ChatterboxTTS.from_pretrained(device="cuda" if torch.cuda.is_available() else "cpu")
        print(f"✅ Model loaded successfully on device: {model.device if hasattr(model, 'device') else 'CPU'}")
        
        # Test text
        text = "Ezreal and Jinx teamed up with Ahri, Yasuo, and Teemo to take down the enemy's Nexus in an epic late-game pentakill."
        print(f"📝 Test text: {text}")
        
        # Generate audio
        print("🎵 Generating audio...")
        wav = model.generate(text)
        print(f"✅ Audio generated successfully! Shape: {wav.shape if hasattr(wav, 'shape') else 'Unknown'}")
        
        # Save audio
        output_file = "test_chatterbox.wav"
        ta.save(output_file, wav, model.sr)
        print(f"💾 Audio saved to: {output_file}")
        
        print("\n🎉 Chatterbox TTS test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error during test: {e}")
        return False

if __name__ == "__main__":
    test_chatterbox_tts()
