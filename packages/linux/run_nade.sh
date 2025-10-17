#!/bin/bash
# NADE Linux Launcher

cd "$(dirname "$0")"

# Add plugins to Python path
export PYTHONPATH="$PWD/plugins:$PYTHONPATH"

# Check dependencies
if ! python3 -c "import sounddevice" 2>/dev/null; then
    echo "Installing Python dependencies..."
    pip3 install --user -r requirements.txt
fi

# Run NADE
./bin/NADE
