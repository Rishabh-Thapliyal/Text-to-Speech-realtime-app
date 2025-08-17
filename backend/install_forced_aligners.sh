#!/bin/bash

echo "🚀 Installing Forced Alignment Tools for High-Quality Character Alignment"
echo "=================================================================="

# Update package manager
echo "📦 Updating package manager..."
if command -v apt-get &> /dev/null; then
    sudo apt-get update
    sudo apt-get install -y python3-dev build-essential cmake
elif command -v brew &> /dev/null; then
    brew update
    brew install cmake
fi

# Install Python dependencies
echo "🐍 Installing Python dependencies..."
pip install --upgrade pip

# Install Montreal Forced Aligner (MFA)
echo "🎯 Installing Montreal Forced Aligner (MFA)..."
pip install montreal-forced-aligner

# Install aeneas (alternative forced aligner)
echo "🔧 Installing aeneas forced aligner..."
pip install aeneas

# Install praatio for TextGrid parsing
echo "📊 Installing praatio for TextGrid parsing..."
pip install praatio

# Install additional dependencies
echo "📚 Installing additional dependencies..."
pip install numpy scipy librosa soundfile

# Download MFA models (English)
echo "🌍 Downloading MFA English models..."
mfa download acoustic english
mfa download dictionary english

echo ""
echo "✅ Installation complete!"
echo ""
echo "🔧 Available Forced Alignment Tools:"
echo "   • Montreal Forced Aligner (MFA) - Primary tool"
echo "   • aeneas - Alternative forced aligner"
echo "   • praatio - TextGrid parsing"
echo ""
echo "📖 Usage:"
echo "   The system will automatically use forced alignment when audio data is available."
echo "   Fallback to MFA text analysis if forced alignment fails."
echo "   Basic alignment as final fallback."
echo ""
echo "🚀 Ready to generate high-quality character alignments!"
