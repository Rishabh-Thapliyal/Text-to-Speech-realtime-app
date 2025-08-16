#!/usr/bin/env python3
"""
Test script for Chatterbox TTS integration
"""

import warnings
import torch
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS

# Suppress specific deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*LoRACompatibleLinear.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.backends.cuda.sdp_kernel.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*past_key_values.*")

def test_chatterbox_tts():
    """Test the Chatterbox TTS model"""
    print("🎤 Testing Chatterbox TTS Integration")
    print("=" * 50)
    
    try:
        # Initialize model
        print("🔧 Initializing Chatterbox TTS model...")
        
        # Use modern torch attention implementation if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🎯 Using device: {device}")
        
        model = ChatterboxTTS.from_pretrained(device=device)
        print(f"✅ Model loaded successfully on device: {device.upper()}")
        
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
