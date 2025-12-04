#!/bin/bash
# Linux setup script for NDA Python plugins

echo "======================================"
echo "NDA Python Plugins Setup (Linux)"
echo "======================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 not found!"
    echo "Please install Python 3:"
    echo ""
    echo "  Fedora/RHEL: sudo dnf install python3 python3-pip"
    echo "  Ubuntu/Debian: sudo apt install python3 python3-pip"
    echo ""
    exit 1
fi

echo "Python found:"
python3 --version
echo ""

# Install requirements
echo "Installing Python dependencies..."
python3 -m pip install --user --upgrade pip
python3 -m pip install --user -r requirements.txt

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Failed to install dependencies"
    echo ""
    echo "Trying system packages..."

    # Try to install via system package manager
    if command -v dnf &> /dev/null; then
        echo "Detected Fedora/RHEL"
        sudo dnf install python3-numpy python3-pyaudio
    elif command -v apt &> /dev/null; then
        echo "Detected Ubuntu/Debian"
        sudo apt install python3-numpy python3-pyaudio
    fi
fi

echo ""
echo "======================================"
echo "Testing plugins..."
echo "======================================"
python3 test_plugins.py

if [ $? -eq 0 ]; then
    echo ""
    echo "======================================"
    echo "Setup completed successfully!"
    echo "======================================"
    echo ""
    echo "You can now:"
    echo "  1. Run ./test_plugins.py to test the plugins"
    echo "  2. Use the plugins from the NDA C++ application"
    echo ""
else
    echo ""
    echo "======================================"
    echo "Setup completed with warnings"
    echo "======================================"
    echo ""
    echo "The basic plugins work, but PyAudio might not be available."
    echo "This means microphone and speaker plugins won't work."
    echo "You can still use: sine_wave_source, null_sink, wav_file_sink"
    echo ""
fi
