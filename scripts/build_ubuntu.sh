#!/usr/bin/env bash
set -euo pipefail

echo "==> Installing Ubuntu build dependencies (requires sudo)..."
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    qt6-base-dev \
    qt6-base-dev-tools \
    libqt6opengl6-dev \
    libssl-dev \
    python3 \
    python3-dev \
    python3-pip \
    python3-numpy \
    portaudio19-dev \
    python3-pyaudio \
    libgl1-mesa-dev \
    git

echo "==> Installing Python plugin dependencies (user scope)..."
if command -v pip3 >/dev/null 2>&1; then
    pip3 install --user --break-system-packages -r requirements.txt
fi

echo "==> Configuring CMake build..."
/usr/bin/cmake -S . -B build

echo "==> Building NDA..."
/usr/bin/cmake --build build -j"$(nproc)"

echo "==> Launching NDA..."
./build/NDA
