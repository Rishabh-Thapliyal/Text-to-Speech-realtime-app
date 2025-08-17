#!/usr/bin/env python3
"""
Simple configuration file to easily switch between TTS models
"""

import requests
import json

# Server configuration
SERVER_URL = "http://localhost:8001"

def switch_to_kokoro():
    """Switch to Kokoro TTS model"""
    print("🟢 Switching to Kokoro TTS model...")
    try:
        response = requests.post(f"{SERVER_URL}/models/switch/kokoro")
        if response.status_code == 200:
            result = response.json()
            print(f"✅ {result['message']}")
            return True
        else:
            print(f"❌ Failed to switch: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def switch_to_chatterbox():
    """Switch to Chatterbox TTS model"""
    print("🔵 Switching to Chatterbox TTS model...")
    try:
        response = requests.post(f"{SERVER_URL}/models/switch/chatterbox")
        if response.status_code == 200:
            result = response.json()
            print(f"✅ {result['message']}")
            return True
        else:
            print(f"❌ Failed to switch: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def show_current_model():
    """Show current TTS model"""
    print("🔍 Checking current model...")
    try:
        response = requests.get(f"{SERVER_URL}/models/current")
        if response.status_code == 200:
            model_info = response.json()
            print(f"🎤 Current model: {model_info['model_type'].upper()}")
            print(f"   Device: {model_info['device']}")
            print(f"   Available models: {model_info['available_models']}")
            return model_info
        else:
            print(f"❌ Failed to get model info: {response.text}")
            return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def show_available_models():
    """Show all available models and their configuration"""
    print("🔍 Getting available models...")
    try:
        response = requests.get(f"{SERVER_URL}/models")
        if response.status_code == 200:
            models_info = response.json()
            print(f"📋 Available models: {models_info['available_models']}")
            print(f"🎤 Current model: {models_info['current_model']['model_type'].upper()}")
            print("\n📖 Model configurations:")
            
            for model_type, config in models_info['configuration'].items():
                print(f"\n   {model_type.upper()}:")
                for key, value in config.items():
                    print(f"     {key}: {value}")
            
            return models_info
        else:
            print(f"❌ Failed to get models: {response.text}")
            return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def main():
    """Main function with interactive menu"""
    print("🎭 TTS Model Configuration Tool")
    print("=" * 40)
    
    while True:
        print("\nOptions:")
        print("1. Show current model")
        print("2. Show all models")
        print("3. Switch to Kokoro")
        print("4. Switch to Chatterbox")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == "1":
            show_current_model()
        elif choice == "2":
            show_available_models()
        elif choice == "3":
            switch_to_kokoro()
        elif choice == "4":
            switch_to_chatterbox()
        elif choice == "5":
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please enter 1-5.")
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()
