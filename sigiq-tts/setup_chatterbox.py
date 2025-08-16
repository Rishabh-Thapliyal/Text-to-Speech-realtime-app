#!/usr/bin/env python3
"""
Setup script for Chatterbox model weights integration with SigIQ TTS System
This script helps download and configure Chatterbox model weights for use with RealtimeTTS
"""

import os
import sys
import argparse
import json
from pathlib import Path

def create_chatterbox_config():
    """Create configuration for Chatterbox model"""
    config = {
        "chatterbox": {
            "use_local_weights": True,
            "weights_path": "./chatterbox_weights",
            "model_type": "realtime_tts",
            "enable_streaming": True,
            "chunk_size": 50
        }
    }
    
    # Update config.py with Chatterbox settings
    config_path = Path("config.py")
    if config_path.exists():
        print("📝 Updating existing config.py with Chatterbox settings...")
        # Read existing config
        with open(config_path, 'r') as f:
            content = f.read()
        
        # Add Chatterbox config if not present
        if "chatterbox" not in content:
            # Find the TTS_CONFIG section and add chatterbox config
            if "TTS_CONFIG = {" in content:
                # Simple replacement - in production you'd want a more robust parser
                chatterbox_config = '''    # Chatterbox specific settings
    "chatterbox": {
        "use_local_weights": True,  # Use local Chatterbox model weights
        "weights_path": "./chatterbox_weights",  # Path to Chatterbox weights
        "model_type": "realtime_tts",  # Model type for RealtimeTTS
        "enable_streaming": True,  # Enable real-time streaming
        "chunk_size": 50,  # Text chunk size for streaming
    }'''
                
                # Insert before the closing brace of TTS_CONFIG
                content = content.replace("    }", f"{chatterbox_config}\n    }}")
                
                with open(config_path, 'w') as f:
                    f.write(content)
                print("✅ Added Chatterbox configuration to config.py")
            else:
                print("⚠️  Could not find TTS_CONFIG section in config.py")
        else:
            print("✅ Chatterbox configuration already exists in config.py")
    else:
        print("❌ config.py not found. Please run this script from the project root directory.")

def download_model_weights(model_name, output_dir):
    """Download model weights from Hugging Face"""
    try:
        from transformers import AutoTokenizer, AutoModelForTextToSpeech
        
        print(f"📥 Downloading model: {model_name}")
        print(f"📁 Output directory: {output_dir}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Download tokenizer and model
        print("🔍 Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained(output_dir)
        
        print("🔍 Downloading model...")
        model = AutoModelForTextToSpeech.from_pretrained(model_name)
        model.save_pretrained(output_dir)
        
        print("✅ Model weights downloaded successfully!")
        
        # Create model info file
        info = {
            "model_name": model_name,
            "model_type": "text_to_speech",
            "framework": "transformers",
            "description": "Chatterbox-compatible TTS model for RealtimeTTS"
        }
        
        with open(os.path.join(output_dir, "model_info.json"), 'w') as f:
            json.dump(info, f, indent=2)
        
        print(f"📋 Model info saved to {output_dir}/model_info.json")
        
    except ImportError:
        print("❌ Transformers library not available. Install with: pip install transformers")
        return False
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        return False
    
    return True

def setup_environment():
    """Set up the Python environment for Chatterbox TTS"""
    print("🔧 Setting up Python environment for Chatterbox TTS...")
    
    # Check if required packages are installed
    required_packages = [
        "torch",
        "torchaudio", 
        "transformers",
        "soundfile",
        "librosa"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} is available")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} is missing")
    
    if missing_packages:
        print(f"\n📦 Installing missing packages: {', '.join(missing_packages)}")
        print("Run the following command:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    else:
        print("✅ All required packages are available")
        return True

def create_chatterbox_weights_dir():
    """Create the Chatterbox weights directory structure"""
    weights_dir = Path("./chatterbox_weights")
    weights_dir.mkdir(exist_ok=True)
    
    # Create a README file
    readme_content = """# Chatterbox Model Weights

This directory contains the Chatterbox model weights for the SigIQ TTS System.

## Structure
- `config.json` - Model configuration
- `pytorch_model.bin` - Model weights
- `tokenizer.json` - Tokenizer configuration
- `model_info.json` - Model metadata

## Usage
The system will automatically load these weights when `use_local_weights` is set to `true` in the configuration.

## Downloading Weights
Use the setup script to download model weights:
```bash
python setup_chatterbox.py --download-model "your/model/name"
```

## Custom Models
To use your own Chatterbox model:
1. Place your model files in this directory
2. Ensure the directory structure matches the expected format
3. Update the configuration if needed
"""
    
    with open(weights_dir / "README.md", 'w') as f:
        f.write(readme_content)
    
    print(f"📁 Created Chatterbox weights directory: {weights_dir}")
    return weights_dir

def main():
    parser = argparse.ArgumentParser(description="Setup Chatterbox model weights for SigIQ TTS System")
    parser.add_argument("--download-model", type=str, help="Hugging Face model name to download")
    parser.add_argument("--setup-config", action="store_true", help="Setup Chatterbox configuration")
    parser.add_argument("--setup-env", action="store_true", help="Check and setup Python environment")
    parser.add_argument("--create-dirs", action="store_true", help="Create necessary directories")
    
    args = parser.parse_args()
    
    print("🎤 SigIQ TTS System - Chatterbox Setup")
    print("=" * 50)
    
    if args.setup_env or not args.download_model:
        if not setup_environment():
            print("\n❌ Environment setup failed. Please install missing packages.")
            return
    
    if args.create_dirs or not args.download_model:
        create_chatterbox_weights_dir()
    
    if args.setup_config or not args.download_model:
        create_chatterbox_config()
    
    if args.download_model:
        weights_dir = "./chatterbox_weights"
        if download_model_weights(args.download_model, weights_dir):
            print(f"\n🎉 Successfully downloaded {args.download_model} to {weights_dir}")
            print("The model is now ready to use with the TTS system!")
        else:
            print(f"\n❌ Failed to download {args.download_model}")
            return
    
    if not args.download_model and not args.setup_config and not args.setup_env and not args.create_dirs:
        # Default behavior - run all setup steps
        print("\n🚀 Running complete setup...")
        create_chatterbox_weights_dir()
        create_chatterbox_config()
        
        print("\n📋 Setup complete! Next steps:")
        print("1. Download your Chatterbox model weights:")
        print("   python setup_chatterbox.py --download-model 'your/model/name'")
        print("2. Start the TTS system:")
        print("   ./start.sh")
        print("3. The system will automatically use your local model weights")

if __name__ == "__main__":
    main()
