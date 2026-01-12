/**
 * NDA v2.1 - Ring Buffer Unit Tests
 *
 * Tests for the lock-free SPSC ring buffer implementation.
 * Validates correctness, thread safety, and overflow/underflow detection.
 */

#include "audio/RingBuffer.h"
#include <iostream>
#include <thread>
#include <chrono>
#include <random>
#include <cassert>
#include <cmath>
#include <atomic>

using namespace nda;

// ============================================================================
// Test Utilities
// ============================================================================

void fillTestPattern(float** data, int channels, int frames, int seed) {
    for (int ch = 0; ch < channels; ++ch) {
        for (int frame = 0; frame < frames; ++frame) {
            data[ch][frame] = static_cast<float>(seed * 1000 + ch * 100000 + frame);
        }
    }
}

bool verifyTestPattern(float** data, int channels, int frames, int seed) {
    for (int ch = 0; ch < channels; ++ch) {
        for (int frame = 0; frame < frames; ++frame) {
            float expected = static_cast<float>(seed * 1000 + ch * 100000 + frame);
            if (std::abs(data[ch][frame] - expected) > 0.001f) {
                std::cerr << "Mismatch at ch=" << ch << " frame=" << frame
                          << " expected=" << expected << " got=" << data[ch][frame] << "\n";
                return false;
            }
        }
    }
    return true;
}

// ============================================================================
// Test 1: Basic Correctness
// ============================================================================

bool test_basic_correctness() {
    std::cout << "Test 1: Basic read/write correctness... ";

    RingBuffer rb;
    if (!rb.initialize(2, 1024)) {
        std::cout << "FAIL (initialize failed)\n";
        return false;
    }

    // Allocate test data
    const int testFrames = 512;
    float writeData[2][512];
    float readData[2][512];

    float* writePtrs[2] = {writeData[0], writeData[1]};
    float* readPtrs[2] = {readData[0], readData[1]};

    // Fill with test pattern
    fillTestPattern(writePtrs, 2, testFrames, 42);

    // Write to buffer
    int written = rb.write(const_cast<const float**>(writePtrs), testFrames);
    if (written != testFrames) {
        std::cout << "FAIL (wrote " << written << " expected " << testFrames << ")\n";
        return false;
    }

    if (rb.getAvailableRead() != testFrames) {
        std::cout << "FAIL (available read " << rb.getAvailableRead() << " expected " << testFrames << ")\n";
        return false;
    }

    // Read from buffer
    int read = rb.read(readPtrs, testFrames);
    if (read != testFrames) {
        std::cout << "FAIL (read " << read << " expected " << testFrames << ")\n";
        return false;
    }

    if (rb.getAvailableRead() != 0) {
        std::cout << "FAIL (available read " << rb.getAvailableRead() << " expected 0)\n";
        return false;
    }

    // Verify data matches
    if (!verifyTestPattern(readPtrs, 2, testFrames, 42)) {
        std::cout << "FAIL (data mismatch)\n";
        return false;
    }

    std::cout << "PASS\n";
    return true;
}

// ============================================================================
// Test 2: Overflow Detection
// ============================================================================

bool test_overflow() {
    std::cout << "Test 2: Overflow detection... ";

    RingBuffer rb;
    if (!rb.initialize(2, 512)) {
        std::cout << "FAIL (initialize failed)\n";
        return false;
    }

    float writeData[2][1024];  // Write more than capacity
    float* writePtrs[2] = {writeData[0], writeData[1]};

    fillTestPattern(writePtrs, 2, 1024, 1);

    int written = rb.write(const_cast<const float**>(writePtrs), 1024);

    // Should not write all (buffer capacity is 512, sentinel uses 1, so max is 511)
    if (written >= 1024) {
        std::cout << "FAIL (wrote all " << written << " frames, expected less)\n";
        return false;
    }

    if (written > 511) {
        std::cout << "FAIL (wrote " << written << " frames, expected <= 511)\n";
        return false;
    }

    // Should detect overflow
    if (rb.getOverruns() == 0) {
        std::cout << "FAIL (no overflow detected)\n";
        return false;
    }

    std::cout << "PASS (wrote " << written << " of 1024 frames, overruns=" << rb.getOverruns() << ")\n";
    return true;
}

// ============================================================================
// Test 3: Underflow Detection
// ============================================================================

bool test_underflow() {
    std::cout << "Test 3: Underflow detection... ";

    RingBuffer rb;
    if (!rb.initialize(2, 1024)) {
        std::cout << "FAIL (initialize failed)\n";
        return false;
    }

    float readData[2][512];
    float* readPtrs[2] = {readData[0], readData[1]};

    // Try to read from empty buffer
    int read = rb.read(readPtrs, 512);

    if (read != 0) {
        std::cout << "FAIL (read " << read << " from empty buffer)\n";
        return false;
    }

    if (rb.getUnderruns() == 0) {
        std::cout << "FAIL (no underrun detected)\n";
        return false;
    }

    std::cout << "PASS (underruns=" << rb.getUnderruns() << ")\n";
    return true;
}

// ============================================================================
// Test 4: Wrap-Around Correctness
// ============================================================================

bool test_wraparound() {
    std::cout << "Test 4: Wrap-around correctness... ";

    RingBuffer rb;
    if (!rb.initialize(2, 1024)) {
        std::cout << "FAIL (initialize failed)\n";
        return false;
    }

    const int chunkSize = 512;
    float writeData[2][512];
    float readData[2][512];

    float* writePtrs[2] = {writeData[0], writeData[1]};
    float* readPtrs[2] = {readData[0], readData[1]};

    // Write and read multiple times to force wrap-around
    for (int iteration = 0; iteration < 20; ++iteration) {
        // Fill with iteration-specific pattern
        fillTestPattern(writePtrs, 2, chunkSize, iteration);

        // Write
        int written = rb.write(const_cast<const float**>(writePtrs), chunkSize);
        if (written != chunkSize) {
            std::cout << "FAIL (iteration " << iteration << " wrote " << written << ")\n";
            return false;
        }

        // Read
        int read = rb.read(readPtrs, chunkSize);
        if (read != chunkSize) {
            std::cout << "FAIL (iteration " << iteration << " read " << read << ")\n";
            return false;
        }

        // Verify data
        if (!verifyTestPattern(readPtrs, 2, chunkSize, iteration)) {
            std::cout << "FAIL (iteration " << iteration << " data mismatch)\n";
            return false;
        }
    }

    std::cout << "PASS\n";
    return true;
}

// ============================================================================
// Test 5: Partial Read/Write
// ============================================================================

bool test_partial_operations() {
    std::cout << "Test 5: Partial read/write... ";

    RingBuffer rb;
    if (!rb.initialize(2, 1024)) {
        std::cout << "FAIL (initialize failed)\n";
        return false;
    }

    float writeData[2][300];
    float readData[2][600];

    float* writePtrs[2] = {writeData[0], writeData[1]};
    float* readPtrs[2] = {readData[0], readData[1]};

    // Write 300 frames
    fillTestPattern(writePtrs, 2, 300, 99);
    int written = rb.write(const_cast<const float**>(writePtrs), 300);
    if (written != 300) {
        std::cout << "FAIL (wrote " << written << " expected 300)\n";
        return false;
    }

    // Try to read 600 frames (should only get 300)
    int read = rb.read(readPtrs, 600);
    if (read != 300) {
        std::cout << "FAIL (read " << read << " expected 300)\n";
        return false;
    }

    if (rb.getUnderruns() == 0) {
        std::cout << "FAIL (no underrun for partial read)\n";
        return false;
    }

    // Verify the 300 frames we did get
    if (!verifyTestPattern(readPtrs, 2, 300, 99)) {
        std::cout << "FAIL (data mismatch)\n";
        return false;
    }

    std::cout << "PASS\n";
    return true;
}

// ============================================================================
// Test 6: Multi-Channel Support
// ============================================================================

bool test_multichannel() {
    std::cout << "Test 6: Multi-channel support (8 channels)... ";

    RingBuffer rb;
    if (!rb.initialize(8, 1024)) {
        std::cout << "FAIL (initialize failed)\n";
        return false;
    }

    const int testFrames = 256;
    float writeData[8][256];
    float readData[8][256];

    float* writePtrs[8];
    float* readPtrs[8];
    for (int ch = 0; ch < 8; ++ch) {
        writePtrs[ch] = writeData[ch];
        readPtrs[ch] = readData[ch];
    }

    // Fill with unique pattern per channel
    for (int ch = 0; ch < 8; ++ch) {
        for (int frame = 0; frame < testFrames; ++frame) {
            writeData[ch][frame] = static_cast<float>(ch * 10000 + frame);
        }
    }

    int written = rb.write(const_cast<const float**>(writePtrs), testFrames);
    if (written != testFrames) {
        std::cout << "FAIL (wrote " << written << ")\n";
        return false;
    }

    int read = rb.read(readPtrs, testFrames);
    if (read != testFrames) {
        std::cout << "FAIL (read " << read << ")\n";
        return false;
    }

    // Verify each channel
    for (int ch = 0; ch < 8; ++ch) {
        for (int frame = 0; frame < testFrames; ++frame) {
            float expected = static_cast<float>(ch * 10000 + frame);
            if (std::abs(readData[ch][frame] - expected) > 0.001f) {
                std::cout << "FAIL (ch " << ch << " frame " << frame << ")\n";
                return false;
            }
        }
    }

    std::cout << "PASS\n";
    return true;
}

// ============================================================================
// Test 7: Clear Operation
// ============================================================================

bool test_clear() {
    std::cout << "Test 7: Clear operation... ";

    RingBuffer rb;
    if (!rb.initialize(2, 1024)) {
        std::cout << "FAIL (initialize failed)\n";
        return false;
    }

    float writeData[2][512];
    float* writePtrs[2] = {writeData[0], writeData[1]};
    fillTestPattern(writePtrs, 2, 512, 1);

    // Write some data
    rb.write(const_cast<const float**>(writePtrs), 512);
    if (rb.getAvailableRead() == 0) {
        std::cout << "FAIL (no data after write)\n";
        return false;
    }

    // Clear the buffer
    rb.clear();

    if (rb.getAvailableRead() != 0) {
        std::cout << "FAIL (data after clear: " << rb.getAvailableRead() << ")\n";
        return false;
    }

    // Write again to verify clear worked
    int written = rb.write(const_cast<const float**>(writePtrs), 512);
    if (written != 512) {
        std::cout << "FAIL (post-clear write " << written << ")\n";
        return false;
    }

    std::cout << "PASS\n";
    return true;
}

// ============================================================================
// Test 8: Multi-Threaded Stress Test
// ============================================================================

bool test_multithreaded() {
    std::cout << "Test 8: Multi-threaded stress test (5 seconds)... ";
    std::cout.flush();

    RingBuffer rb;
    if (!rb.initialize(2, 9600)) {  // 200ms at 48kHz
        std::cout << "FAIL (initialize failed)\n";
        return false;
    }

    std::atomic<bool> running{true};
    std::atomic<uint64_t> totalWritten{0};
    std::atomic<uint64_t> totalRead{0};
    std::atomic<bool> dataCorruption{false};

    // Producer thread (simulates WASAPI capture)
    std::thread producer([&]() {
        const int chunkSize = 480;  // 10ms chunks
        float writeData[2][480];
        float* writePtrs[2] = {writeData[0], writeData[1]};

        uint64_t iteration = 0;

        while (running.load(std::memory_order_relaxed)) {
            // Fill with simple incrementing pattern
            for (int ch = 0; ch < 2; ++ch) {
                for (int frame = 0; frame < chunkSize; ++frame) {
                    writeData[ch][frame] = static_cast<float>((iteration * chunkSize + frame) % 100000);
                }
            }

            // Write to buffer
            int written = rb.write(const_cast<const float**>(writePtrs), chunkSize);
            totalWritten.fetch_add(written, std::memory_order_relaxed);

            ++iteration;

            // Simulate 10ms WASAPI packet interval
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    });

    // Consumer thread (simulates pipeline)
    std::thread consumer([&]() {
        const int chunkSize = 512;  // ~10.67ms chunks (pipeline typical)
        float readData[2][512];
        float* readPtrs[2] = {readData[0], readData[1]};

        while (running.load(std::memory_order_relaxed)) {
            // Read from buffer
            int read = rb.read(readPtrs, chunkSize);
            totalRead.fetch_add(read, std::memory_order_relaxed);

            // Verify data if we got any (check for NaN/Inf)
            if (read > 0) {
                for (int ch = 0; ch < 2; ++ch) {
                    for (int frame = 0; frame < read; ++frame) {
                        float sample = readPtrs[ch][frame];
                        if (std::isnan(sample) || std::isinf(sample)) {
                            dataCorruption.store(true, std::memory_order_relaxed);
                            std::cerr << "CORRUPTION: NaN or Inf detected!\n";
                        }
                    }
                }
            }

            // Simulate ~10.67ms pipeline processing interval
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    });

    // Run for 5 seconds
    std::this_thread::sleep_for(std::chrono::seconds(5));
    running.store(false, std::memory_order_relaxed);

    producer.join();
    consumer.join();

    // Check results
    if (dataCorruption.load()) {
        std::cout << "FAIL (data corruption detected)\n";
        return false;
    }

    uint64_t written = totalWritten.load();
    uint64_t read = totalRead.load();

    std::cout << "PASS\n";
    std::cout << "  Total written: " << written << " frames\n";
    std::cout << "  Total read: " << read << " frames\n";
    std::cout << "  Overruns: " << rb.getOverruns() << "\n";
    std::cout << "  Underruns: " << rb.getUnderruns() << "\n";

    return true;
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "========================================\n";
    std::cout << "NDA v2.1 Ring Buffer Unit Tests\n";
    std::cout << "========================================\n\n";

    int passed = 0;
    int failed = 0;

    if (test_basic_correctness()) ++passed; else ++failed;
    if (test_overflow()) ++passed; else ++failed;
    if (test_underflow()) ++passed; else ++failed;
    if (test_wraparound()) ++passed; else ++failed;
    if (test_partial_operations()) ++passed; else ++failed;
    if (test_multichannel()) ++passed; else ++failed;
    if (test_clear()) ++passed; else ++failed;
    if (test_multithreaded()) ++passed; else ++failed;

    std::cout << "\n========================================\n";
    std::cout << "Results: " << passed << " passed, " << failed << " failed\n";
    std::cout << "========================================\n";

    return (failed == 0) ? 0 : 1;
}
