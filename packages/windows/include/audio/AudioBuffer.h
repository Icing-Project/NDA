#ifndef AUDIOBUFFER_H
#define AUDIOBUFFER_H

#include <vector>
#include <cstdint>
#include <algorithm>

class AudioBuffer
{
public:
    inline AudioBuffer(int channels = 2, int frames = 512)
        : channels_(channels), frames_(frames)
    {
        data_.resize(channels);
        for (auto& channel : data_) {
            channel.resize(frames, 0.0f);
        }
    }

    inline ~AudioBuffer() {}

    inline void resize(int channels, int frames)
    {
        channels_ = channels;
        frames_ = frames;
        data_.resize(channels);
        for (auto& channel : data_) {
            channel.resize(frames, 0.0f);
        }
    }

    inline void clear()
    {
        for (auto& channel : data_) {
            std::fill(channel.begin(), channel.end(), 0.0f);
        }
    }

    inline float* getChannelData(int channel)
    {
        if (channel >= 0 && channel < channels_) {
            return data_[channel].data();
        }
        return nullptr;
    }

    inline const float* getChannelData(int channel) const
    {
        if (channel >= 0 && channel < channels_) {
            return data_[channel].data();
        }
        return nullptr;
    }

    inline int getChannelCount() const { return channels_; }
    inline int getFrameCount() const { return frames_; }

    inline void copyFrom(const AudioBuffer& other)
    {
        if (channels_ != other.channels_ || frames_ != other.frames_) {
            resize(other.channels_, other.frames_);
        }

        for (int ch = 0; ch < channels_; ++ch) {
            std::copy(other.data_[ch].begin(), other.data_[ch].end(), data_[ch].begin());
        }
    }

    inline void mixWith(const AudioBuffer& other, float gain = 1.0f)
    {
        int minChannels = std::min(channels_, other.channels_);
        int minFrames = std::min(frames_, other.frames_);

        for (int ch = 0; ch < minChannels; ++ch) {
            for (int frame = 0; frame < minFrames; ++frame) {
                data_[ch][frame] += other.data_[ch][frame] * gain;
            }
        }
    }

private:
    int channels_;
    int frames_;
    std::vector<std::vector<float>> data_;
};

#endif // AUDIOBUFFER_H
