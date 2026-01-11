#ifndef RINGBUFFER_H
#define RINGBUFFER_H

#include <atomic>
#include <vector>
#include <cstdint>
#include <algorithm>
#include <iostream>

namespace nda {

/**
 * @brief Lock-free single-producer-single-consumer (SPSC) ring buffer for planar float32 audio.
 *
 * Thread-safe for one writer thread and one reader thread. Uses atomic read/write pointers
 * with proper memory ordering to ensure correct synchronization without locks.
 *
 * Storage: Planar format (separate circular buffer per channel)
 * Capacity: Fixed at initialization (cannot be resized after initialize())
 *
 * NOTE: This is a header-only implementation to allow use by plugins (DLLs).
 */
class RingBuffer {
public:
    RingBuffer()
        : channels_(0)
        , capacity_(0)
        , readPos_(0)
        , writePos_(0)
        , overruns_(0)
        , underruns_(0)
    {
    }

    ~RingBuffer() {
        // Vectors automatically cleaned up
    }

    /**
     * @brief Initialize buffer with channel count and capacity.
     */
    bool initialize(int channels, int capacityFrames) {
        if (channels <= 0 || channels > 8) {
            std::cerr << "[RingBuffer] Invalid channel count: " << channels << "\n";
            return false;
        }

        if (capacityFrames <= 0) {
            std::cerr << "[RingBuffer] Invalid capacity: " << capacityFrames << "\n";
            return false;
        }

        channels_ = channels;
        capacity_ = capacityFrames;

        // Allocate planar storage
        buffer_.resize(channels_);
        for (int ch = 0; ch < channels_; ++ch) {
            buffer_[ch].resize(capacity_, 0.0f);
        }

        // Initialize pointers
        readPos_.store(0, std::memory_order_relaxed);
        writePos_.store(0, std::memory_order_relaxed);

        // Reset counters
        overruns_.store(0, std::memory_order_relaxed);
        underruns_.store(0, std::memory_order_relaxed);

        return true;
    }

    /**
     * @brief Write audio to buffer (called by producer thread).
     */
    int write(const float* const* channelData, int frameCount) {
        if (frameCount <= 0) {
            return 0;
        }

        // Load current positions
        int currentRead = readPos_.load(std::memory_order_acquire);
        int currentWrite = writePos_.load(std::memory_order_relaxed);

        // Calculate available space
        int available = getAvailableReadInternal(currentRead, currentWrite);
        int freeSpace = capacity_ - available - 1;  // -1 for sentinel

        if (freeSpace <= 0) {
            overruns_.fetch_add(1, std::memory_order_relaxed);
            return 0;
        }

        int framesToWrite = std::min(frameCount, freeSpace);
        int endSpace = capacity_ - currentWrite;

        if (framesToWrite <= endSpace) {
            for (int ch = 0; ch < channels_; ++ch) {
                std::copy(channelData[ch],
                          channelData[ch] + framesToWrite,
                          buffer_[ch].data() + currentWrite);
            }
        } else {
            int firstPart = endSpace;
            for (int ch = 0; ch < channels_; ++ch) {
                std::copy(channelData[ch],
                          channelData[ch] + firstPart,
                          buffer_[ch].data() + currentWrite);
                std::copy(channelData[ch] + firstPart,
                          channelData[ch] + framesToWrite,
                          buffer_[ch].data());
            }
        }

        int newWrite = (currentWrite + framesToWrite) % capacity_;
        writePos_.store(newWrite, std::memory_order_release);

        if (framesToWrite < frameCount) {
            overruns_.fetch_add(1, std::memory_order_relaxed);
        }

        return framesToWrite;
    }

    /**
     * @brief Read audio from buffer (called by consumer thread).
     */
    int read(float** channelData, int frameCount) {
        if (frameCount <= 0) {
            return 0;
        }

        int currentWrite = writePos_.load(std::memory_order_acquire);
        int currentRead = readPos_.load(std::memory_order_relaxed);

        int available = getAvailableReadInternal(currentRead, currentWrite);

        if (available <= 0) {
            underruns_.fetch_add(1, std::memory_order_relaxed);
            return 0;
        }

        int framesToRead = std::min(frameCount, available);
        int endSpace = capacity_ - currentRead;

        if (framesToRead <= endSpace) {
            for (int ch = 0; ch < channels_; ++ch) {
                std::copy(buffer_[ch].data() + currentRead,
                          buffer_[ch].data() + currentRead + framesToRead,
                          channelData[ch]);
            }
        } else {
            int firstPart = endSpace;
            int secondPart = framesToRead - endSpace;
            for (int ch = 0; ch < channels_; ++ch) {
                std::copy(buffer_[ch].data() + currentRead,
                          buffer_[ch].data() + capacity_,
                          channelData[ch]);
                std::copy(buffer_[ch].data(),
                          buffer_[ch].data() + secondPart,
                          channelData[ch] + firstPart);
            }
        }

        int newRead = (currentRead + framesToRead) % capacity_;
        readPos_.store(newRead, std::memory_order_release);

        if (framesToRead < frameCount) {
            underruns_.fetch_add(1, std::memory_order_relaxed);
        }

        return framesToRead;
    }

    /**
     * @brief Query number of frames available for reading.
     */
    int getAvailableRead() const {
        int currentRead = readPos_.load(std::memory_order_relaxed);
        int currentWrite = writePos_.load(std::memory_order_relaxed);
        return getAvailableReadInternal(currentRead, currentWrite);
    }

    /**
     * @brief Query number of frames available for writing.
     */
    int getAvailableWrite() const {
        int available = getAvailableRead();
        return capacity_ - available - 1;
    }

    int getCapacity() const { return capacity_; }
    int getChannels() const { return channels_; }

    /**
     * @brief Clear buffer (reset to empty state).
     */
    void clear() {
        readPos_.store(0, std::memory_order_relaxed);
        writePos_.store(0, std::memory_order_relaxed);
        for (int ch = 0; ch < channels_; ++ch) {
            std::fill(buffer_[ch].begin(), buffer_[ch].end(), 0.0f);
        }
    }

    uint64_t getOverruns() const { return overruns_.load(std::memory_order_relaxed); }
    uint64_t getUnderruns() const { return underruns_.load(std::memory_order_relaxed); }

private:
    int channels_;
    int capacity_;
    std::vector<std::vector<float>> buffer_;
    std::atomic<int> readPos_;
    std::atomic<int> writePos_;
    std::atomic<uint64_t> overruns_;
    std::atomic<uint64_t> underruns_;

    int getAvailableReadInternal(int readPos, int writePos) const {
        if (writePos >= readPos) {
            return writePos - readPos;
        } else {
            return capacity_ - readPos + writePos;
        }
    }
};

} // namespace nda

#endif // RINGBUFFER_H
