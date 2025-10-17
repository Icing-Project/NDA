/**
 * @file python_plugin_example.cpp
 * @brief Example demonstrating how to use Python plugins in NADE
 *
 * This example shows:
 * 1. Loading Python plugins from C++
 * 2. Using Python audio sources and sinks
 * 3. Processing audio through the Python plugin pipeline
 */

#include <iostream>
#include <chrono>
#include <thread>

#ifdef NADE_ENABLE_PYTHON
#include "plugins/PythonPluginBridge.h"
#endif

#include "audio/AudioBuffer.h"

using namespace NADE;

void exampleSineWaveToNullSink() {
    std::cout << "============================================" << std::endl;
    std::cout << "Python Plugin Example: Sine Wave -> Null Sink" << std::endl;
    std::cout << "============================================" << std::endl;

#ifdef NADE_ENABLE_PYTHON
    // Create Python plugin bridges
    PythonPluginBridge* sineWave = PythonPluginFactory::createPlugin("sine_wave_source");
    PythonPluginBridge* nullSink = PythonPluginFactory::createPlugin("null_sink");

    if (!sineWave || !nullSink) {
        std::cerr << "Failed to load Python plugins" << std::endl;
        delete sineWave;
        delete nullSink;
        return;
    }

    // Get plugin info
    PluginInfo sineInfo = sineWave->getInfo();
    PluginInfo nullInfo = nullSink->getInfo();

    std::cout << "Loaded: " << sineInfo.name << " v" << sineInfo.version << std::endl;
    std::cout << "Loaded: " << nullInfo.name << " v" << nullInfo.version << std::endl;

    // Initialize plugins
    if (!sineWave->initialize() || !nullSink->initialize()) {
        std::cerr << "Failed to initialize plugins" << std::endl;
        delete sineWave;
        delete nullSink;
        return;
    }

    // Configure plugins
    sineWave->setParameter("frequency", "440");  // A4 note

    // Start plugins
    if (!sineWave->start() || !nullSink->start()) {
        std::cerr << "Failed to start plugins" << std::endl;
        delete sineWave;
        delete nullSink;
        return;
    }

    // Process audio for 3 seconds
    const int sampleRate = 48000;
    const int bufferSize = 512;
    const int channels = 2;
    const double duration = 3.0;

    AudioBuffer buffer(channels, bufferSize);

    int totalFrames = static_cast<int>(sampleRate * duration);
    int framesProcessed = 0;

    std::cout << "Processing audio for " << duration << " seconds..." << std::endl;

    while (framesProcessed < totalFrames) {
        // Read from sine wave generator
        if (!sineWave->readAudio(buffer)) {
            std::cerr << "Failed to read audio" << std::endl;
            break;
        }

        // Write to null sink
        if (!nullSink->writeAudio(buffer)) {
            std::cerr << "Failed to write audio" << std::endl;
            break;
        }

        framesProcessed += bufferSize;

        // Sleep to simulate real-time processing
        std::this_thread::sleep_for(
            std::chrono::microseconds(bufferSize * 1000000 / sampleRate)
        );
    }

    std::cout << "Processing complete!" << std::endl;

    // Stop and cleanup
    sineWave->stop();
    nullSink->stop();

    delete sineWave;
    delete nullSink;

#else
    std::cout << "Python support not enabled!" << std::endl;
    std::cout << "Rebuild with -DNADE_ENABLE_PYTHON=ON" << std::endl;
#endif
}

void exampleSineWaveToWavFile() {
    std::cout << "\n============================================" << std::endl;
    std::cout << "Python Plugin Example: Sine Wave -> WAV File" << std::endl;
    std::cout << "============================================" << std::endl;

#ifdef NADE_ENABLE_PYTHON
    // Create Python plugin bridges
    PythonPluginBridge* sineWave = PythonPluginFactory::createPlugin("sine_wave_source");
    PythonPluginBridge* wavFile = PythonPluginFactory::createPlugin("wav_file_sink");

    if (!sineWave || !wavFile) {
        std::cerr << "Failed to load Python plugins" << std::endl;
        delete sineWave;
        delete wavFile;
        return;
    }

    // Initialize and start plugins
    sineWave->initialize();
    wavFile->initialize();

    sineWave->setParameter("frequency", "440");
    wavFile->setParameter("filename", "python_test.wav");

    sineWave->start();
    wavFile->start();

    // Process audio for 2 seconds
    const int sampleRate = 48000;
    const int bufferSize = 512;
    const int channels = 2;
    const double duration = 2.0;

    AudioBuffer buffer(channels, bufferSize);

    int totalFrames = static_cast<int>(sampleRate * duration);
    int framesProcessed = 0;

    std::cout << "Recording to file for " << duration << " seconds..." << std::endl;

    while (framesProcessed < totalFrames) {
        sineWave->readAudio(buffer);
        wavFile->writeAudio(buffer);
        framesProcessed += bufferSize;
    }

    std::cout << "Recording complete!" << std::endl;

    // Stop and cleanup
    sineWave->stop();
    wavFile->stop();

    delete sineWave;
    delete wavFile;

#else
    std::cout << "Python support not enabled!" << std::endl;
#endif
}

void listPythonPlugins() {
    std::cout << "\n============================================" << std::endl;
    std::cout << "Available Python Plugins" << std::endl;
    std::cout << "============================================" << std::endl;

#ifdef NADE_ENABLE_PYTHON
    const char* pluginNames[] = {
        "sine_wave_source",
        "null_sink",
        "wav_file_sink",
        "pulseaudio_microphone",
        "pulseaudio_speaker"
    };

    for (const char* name : pluginNames) {
        PythonPluginBridge* plugin = PythonPluginFactory::createPlugin(name);
        if (plugin) {
            PluginInfo info = plugin->getInfo();
            std::cout << "\n" << info.name << " v" << info.version << std::endl;
            std::cout << "  Type: ";
            switch (info.type) {
                case PluginType::AudioSource: std::cout << "Audio Source"; break;
                case PluginType::AudioSink: std::cout << "Audio Sink"; break;
                default: std::cout << "Unknown"; break;
            }
            std::cout << std::endl;
            std::cout << "  Description: " << info.description << std::endl;
            std::cout << "  Author: " << info.author << std::endl;
            delete plugin;
        }
    }
#else
    std::cout << "Python support not enabled!" << std::endl;
#endif
}

int main(int argc, char* argv[]) {
    std::cout << "NADE Python Plugin Example" << std::endl;
    std::cout << "==========================" << std::endl;

    // List available plugins
    listPythonPlugins();

    // Run examples
    exampleSineWaveToNullSink();
    exampleSineWaveToWavFile();

    std::cout << "\nAll examples completed!" << std::endl;

    return 0;
}
