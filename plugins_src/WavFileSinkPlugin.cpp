#include "plugins/AudioSinkPlugin.h"
#include <fstream>
#include <cstring>
#include <cstdint>
#include <iostream>
#include <chrono>

namespace NADE {

class WavFileSinkPlugin : public AudioSinkPlugin {
public:
    WavFileSinkPlugin()
        : state_(PluginState::Unloaded),
          sampleRate_(48000),
          channels_(2),
          bufferSize_(512),
          totalFrames_(0),
          file_(nullptr)
    {
    }

    ~WavFileSinkPlugin() override {
        if (state_ != PluginState::Unloaded) {
            shutdown();
        }
    }

    bool initialize() override {
        state_ = PluginState::Initialized;
        return true;
    }

    void shutdown() override {
        stop();
        state_ = PluginState::Unloaded;
    }

    bool start() override {
        if (state_ != PluginState::Initialized) return false;

        // Generate filename with timestamp
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        std::tm tm = *std::localtime(&time_t);

        char filename[256];
        std::snprintf(filename, sizeof(filename),
                     "recording_%04d%02d%02d_%02d%02d%02d.wav",
                     tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday,
                     tm.tm_hour, tm.tm_min, tm.tm_sec);

        currentFilename_ = filename;

        // Open file
        file_ = fopen(currentFilename_.c_str(), "wb");
        if (!file_) {
            std::cerr << "[WavFileSink] Failed to open file: " << currentFilename_ << std::endl;
            return false;
        }

        // Write WAV header (placeholder, will be updated in stop())
        writeWavHeader(0);

        totalFrames_ = 0;
        state_ = PluginState::Running;

        std::cout << "[WavFileSink] Recording to: " << currentFilename_ << std::endl;
        std::cout << "[WavFileSink] Format: " << sampleRate_ << " Hz, "
                  << channels_ << " channels, 32-bit float" << std::endl;

        return true;
    }

    void stop() override {
        if (state_ != PluginState::Running) {
            return; // Already stopped
        }

        if (file_) {
            // Update WAV header with correct size
            fseek(file_, 0, SEEK_SET);
            writeWavHeader(totalFrames_);
            fflush(file_);
            fclose(file_);
            file_ = nullptr;

            double duration = static_cast<double>(totalFrames_) / sampleRate_;
            std::cout << "[WavFileSink] Recording stopped" << std::endl;
            std::cout << "[WavFileSink] Saved " << totalFrames_ << " frames ("
                      << duration << " seconds) to " << currentFilename_ << std::endl;
        }

        state_ = PluginState::Initialized;
    }

    PluginInfo getInfo() const override {
        return {
            "WAV File Recorder",
            "1.0.0",
            "NADE Team",
            "Records audio to WAV file (32-bit float PCM)",
            PluginType::AudioSink,
            NADE_PLUGIN_API_VERSION
        };
    }

    PluginState getState() const override {
        return state_;
    }

    void setParameter(const std::string& key, const std::string& value) override {
        if (key == "filename") {
            customFilename_ = value;
        }
    }

    std::string getParameter(const std::string& key) const override {
        if (key == "filename") return currentFilename_;
        if (key == "sampleRate") return std::to_string(sampleRate_);
        if (key == "channels") return std::to_string(channels_);
        return "";
    }

    bool writeAudio(const AudioBuffer& buffer) override {
        if (!file_ || state_ != PluginState::Running) {
            return false;
        }

        int frameCount = buffer.getFrameCount();
        int channelCount = buffer.getChannelCount();

        // Interleave samples and write to file
        std::vector<float> interleavedData(frameCount * channelCount);

        for (int frame = 0; frame < frameCount; ++frame) {
            for (int ch = 0; ch < channelCount; ++ch) {
                const float* channelData = buffer.getChannelData(ch);
                interleavedData[frame * channelCount + ch] = channelData[frame];
            }
        }

        // Write to file
        size_t written = fwrite(interleavedData.data(), sizeof(float),
                                frameCount * channelCount, file_);

        if (written != static_cast<size_t>(frameCount * channelCount)) {
            std::cerr << "[WavFileSink] Write error!" << std::endl;
            return false;
        }

        totalFrames_ += frameCount;

        // Print progress every second
        if (totalFrames_ % sampleRate_ < frameCount) {
            double seconds = static_cast<double>(totalFrames_) / sampleRate_;
            std::cout << "[WavFileSink] Recording: " << seconds << "s" << std::endl;
        }

        return true;
    }

    int getSampleRate() const override { return sampleRate_; }
    int getChannels() const override { return channels_; }
    int getBufferSize() const override { return bufferSize_; }
    int getAvailableSpace() const override { return 1000000; } // Plenty of space

    void setSampleRate(int sampleRate) override {
        if (state_ == PluginState::Unloaded || state_ == PluginState::Initialized) {
            sampleRate_ = sampleRate;
        }
    }

    void setChannels(int channels) override {
        if (state_ == PluginState::Unloaded || state_ == PluginState::Initialized) {
            channels_ = channels;
        }
    }

    void setBufferSize(int samples) override {
        if (state_ == PluginState::Unloaded || state_ == PluginState::Initialized) {
            bufferSize_ = samples;
        }
    }

private:
    void writeWavHeader(uint32_t totalFrames) {
        // Calculate sizes
        uint32_t dataSize = totalFrames * channels_ * sizeof(float);
        uint32_t fileSize = 36 + dataSize;

        // RIFF header
        fwrite("RIFF", 1, 4, file_);
        fwrite(&fileSize, 4, 1, file_);
        fwrite("WAVE", 1, 4, file_);

        // fmt chunk
        fwrite("fmt ", 1, 4, file_);
        uint32_t fmtSize = 16;
        fwrite(&fmtSize, 4, 1, file_);

        uint16_t audioFormat = 3; // 3 = IEEE float
        uint16_t numChannels = channels_;
        uint32_t sampleRateVal = sampleRate_;
        uint32_t byteRate = sampleRate_ * channels_ * sizeof(float);
        uint16_t blockAlign = channels_ * sizeof(float);
        uint16_t bitsPerSample = 32;

        fwrite(&audioFormat, 2, 1, file_);
        fwrite(&numChannels, 2, 1, file_);
        fwrite(&sampleRateVal, 4, 1, file_);
        fwrite(&byteRate, 4, 1, file_);
        fwrite(&blockAlign, 2, 1, file_);
        fwrite(&bitsPerSample, 2, 1, file_);

        // data chunk
        fwrite("data", 1, 4, file_);
        fwrite(&dataSize, 4, 1, file_);
    }

    PluginState state_;
    int sampleRate_;
    int channels_;
    int bufferSize_;
    uint32_t totalFrames_;
    FILE* file_;
    std::string currentFilename_;
    std::string customFilename_;
};

} // namespace NADE

// Export the plugin
NADE_DECLARE_PLUGIN(NADE::WavFileSinkPlugin)
