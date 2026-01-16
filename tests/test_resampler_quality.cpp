/**
 * NDA v2.0 - Resampler Quality Tests
 * 
 * Tests for sample rate conversion quality and correctness.
 * Validates that resampling preserves audio integrity.
 */

#include "audio/Resampler.h"
#include "audio/AudioBuffer.h"
#include <cmath>
#include <iostream>
#include <algorithm>
#include <cassert>

using namespace nda;

// Helper: Generate sine wave
void generateSine(AudioBuffer& buffer, float frequency, int sampleRate) {
    for (int ch = 0; ch < buffer.getChannelCount(); ++ch) {
        float* data = buffer.getChannelData(ch);
        for (int f = 0; f < buffer.getFrameCount(); ++f) {
            data[f] = std::sin(2.0f * M_PI * frequency * f / sampleRate);
        }
    }
}

// Helper: Measure peak amplitude
float measurePeak(const AudioBuffer& buffer) {
    float maxVal = 0.0f;
    for (int ch = 0; ch < buffer.getChannelCount(); ++ch) {
        const float* data = buffer.getChannelData(ch);
        for (int f = 0; f < buffer.getFrameCount(); ++f) {
            maxVal = std::max(maxVal, std::abs(data[f]));
        }
    }
    return maxVal;
}

// Test 1: Passthrough (no resampling when rates match)
bool testPassthrough() {
    std::cout << "Test 1: Passthrough (48kHz → 48kHz)... ";
    
    Resampler r;
    r.initialize(48000, 48000, 2);
    
    AudioBuffer buffer(2, 512);
    generateSine(buffer, 1000.0f, 48000);
    
    AudioBuffer original(2, 512);
    for (int ch = 0; ch < 2; ++ch) {
        std::copy_n(buffer.getChannelData(ch), 512, original.getChannelData(ch));
    }
    
    r.process(buffer);
    
    // Should be unchanged
    if (buffer.getFrameCount() != 512) {
        std::cout << "FAIL (frame count changed: " << buffer.getFrameCount() << ")\n";
        return false;
    }
    
    // Data should be identical
    for (int ch = 0; ch < 2; ++ch) {
        for (int f = 0; f < 512; ++f) {
            if (std::abs(buffer.getChannelData(ch)[f] - original.getChannelData(ch)[f]) > 1e-6f) {
                std::cout << "FAIL (data changed at ch=" << ch << ", f=" << f << ")\n";
                return false;
            }
        }
    }
    
    std::cout << "PASS\n";
    return true;
}

// Test 2: Upsample 44.1kHz → 48kHz
bool testUpsample441to48() {
    std::cout << "Test 2: Upsample (44.1kHz → 48kHz)... ";
    
    Resampler r;
    r.initialize(44100, 48000, 2);
    
    // 10ms of audio at 44.1kHz = 441 samples
    AudioBuffer buffer(2, 441);
    generateSine(buffer, 1000.0f, 44100);
    
    float peakBefore = measurePeak(buffer);
    
    r.process(buffer);
    
    // Should produce ~480 samples (48000 * 0.01)
    int expectedFrames = static_cast<int>(std::ceil(441 * 48000.0 / 44100.0));
    if (std::abs(buffer.getFrameCount() - expectedFrames) > 2) {
        std::cout << "FAIL (expected ~" << expectedFrames 
                  << " frames, got " << buffer.getFrameCount() << ")\n";
        return false;
    }
    
    // Peak amplitude should be preserved (within tolerance)
    float peakAfter = measurePeak(buffer);
    if (std::abs(peakAfter - peakBefore) > 0.1f) {
        std::cout << "FAIL (amplitude changed: " << peakBefore 
                  << " → " << peakAfter << ")\n";
        return false;
    }
    
    std::cout << "PASS (441 → " << buffer.getFrameCount() 
              << " frames, peak preserved)\n";
    return true;
}

// Test 3: Downsample 96kHz → 48kHz
bool testDownsample96to48() {
    std::cout << "Test 3: Downsample (96kHz → 48kHz)... ";
    
    Resampler r;
    r.initialize(96000, 48000, 2);
    
    // 10ms of audio at 96kHz = 960 samples
    AudioBuffer buffer(2, 960);
    generateSine(buffer, 1000.0f, 96000);
    
    float peakBefore = measurePeak(buffer);
    
    r.process(buffer);
    
    // Should produce exactly 480 samples (2:1 ratio)
    if (buffer.getFrameCount() != 480) {
        std::cout << "FAIL (expected 480 frames, got " 
                  << buffer.getFrameCount() << ")\n";
        return false;
    }
    
    // Peak amplitude should be preserved
    float peakAfter = measurePeak(buffer);
    if (std::abs(peakAfter - peakBefore) > 0.1f) {
        std::cout << "FAIL (amplitude changed: " << peakBefore 
                  << " → " << peakAfter << ")\n";
        return false;
    }
    
    std::cout << "PASS (960 → 480 frames, peak preserved)\n";
    return true;
}

// Test 4: Quality modes comparison
bool testQualityModes() {
    std::cout << "Test 4: Quality modes (Simple/Medium/High)... ";
    
    // Generate 1kHz sine at 44.1kHz
    AudioBuffer input(1, 4410);  // 100ms
    generateSine(input, 1000.0f, 44100);
    
    float peakInput = measurePeak(input);
    
    // Test each quality mode
    for (auto quality : {ResampleQuality::Simple, ResampleQuality::Medium, ResampleQuality::High}) {
        Resampler r;
        r.initialize(44100, 48000, 1, quality);
        
        AudioBuffer buffer(1, 4410);
        std::copy_n(input.getChannelData(0), 4410, buffer.getChannelData(0));
        
        r.process(buffer);
        
        float peakOutput = measurePeak(buffer);
        
        // All quality modes should preserve amplitude reasonably
        if (std::abs(peakOutput - peakInput) > 0.2f) {
            std::cout << "FAIL (quality mode " << static_cast<int>(quality)
                      << " distorted signal: " << peakInput << " → " << peakOutput << ")\n";
            return false;
        }
    }
    
    std::cout << "PASS (all quality modes preserve signal)\n";
    return true;
}

// Main test runner
int main() {
    std::cout << "\n";
    std::cout << "========================================\n";
    std::cout << "NDA v2.0 Resampler Quality Tests\n";
    std::cout << "========================================\n\n";
    
    int passed = 0;
    int failed = 0;
    
    if (testPassthrough()) passed++; else failed++;
    if (testUpsample441to48()) passed++; else failed++;
    if (testDownsample96to48()) passed++; else failed++;
    if (testQualityModes()) passed++; else failed++;
    
    std::cout << "\n========================================\n";
    std::cout << "Results: " << passed << " passed, " << failed << " failed\n";
    std::cout << "========================================\n\n";
    
    if (failed == 0) {
        std::cout << "✓ All resampler tests PASSED!\n\n";
        std::cout << "Validation:\n";
        std::cout << "- Passthrough preserves data exactly\n";
        std::cout << "- Upsampling produces correct frame count\n";
        std::cout << "- Downsampling produces correct frame count\n";
        std::cout << "- All quality modes preserve signal amplitude\n";
        std::cout << "\nResampler is ready for production use.\n";
        return 0;
    } else {
        std::cout << "✗ Some tests FAILED - review resampler implementation\n";
        return 1;
    }
}

// Compilation:
// g++ -std=c++17 tests/test_resampler_quality.cpp src/audio/Resampler.cpp src/audio/AudioBuffer.cpp \
//     -I include/ -o tests/test_resampler_quality

// Run:
// ./tests/test_resampler_quality

