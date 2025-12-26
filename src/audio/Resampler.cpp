#include "audio/Resampler.h"
#include <cstring>
#include <algorithm>
#include <cmath>
#include <iostream>

#ifdef HAVE_LIBSAMPLERATE
#include <samplerate.h>
#endif

namespace nda {

Resampler::Resampler()
    : inputRate_(0)
    , outputRate_(0)
    , channels_(0)
    , quality_(ResampleQuality::Simple)
#ifdef HAVE_LIBSAMPLERATE
    , srcState_(nullptr)
#endif
{
}

Resampler::~Resampler() {
#ifdef HAVE_LIBSAMPLERATE
    if (srcState_) {
        src_delete(static_cast<SRC_STATE*>(srcState_));
        srcState_ = nullptr;
    }
#endif
}

void Resampler::initialize(int inputRate, int outputRate, int channels,
                          ResampleQuality quality) {
    inputRate_ = inputRate;
    outputRate_ = outputRate;
    channels_ = channels;
    quality_ = quality;
    
    if (!isActive()) {
        std::cout << "[Resampler] Passthrough mode (rates match: " << inputRate << "Hz)" << std::endl;
        return;
    }
    
    std::cout << "[Resampler] Initializing " << inputRate << "Hz → " << outputRate << "Hz ("
              << channels << " channels, quality: ";
    
    switch (quality_) {
        case ResampleQuality::Simple:
            std::cout << "Simple)";
            break;
        case ResampleQuality::Medium:
            std::cout << "Medium)";
            break;
        case ResampleQuality::High:
            std::cout << "High)";
            break;
    }
    std::cout << std::endl;
    
    // Initialize continuity buffer (for smooth transitions between buffers)
    lastSamples_.resize(channels_, 0.0f);
    
#ifdef HAVE_LIBSAMPLERATE
    if (quality_ == ResampleQuality::High) {
        int error;
        srcState_ = src_new(SRC_SINC_BEST_QUALITY, channels_, &error);
        if (!srcState_) {
            std::cerr << "[Resampler] libsamplerate init failed (error: " << error
                      << "), falling back to Medium quality" << std::endl;
            quality_ = ResampleQuality::Medium;
        } else {
            std::cout << "[Resampler] Using libsamplerate (SRC_SINC_BEST_QUALITY)" << std::endl;
        }
    }
#else
    if (quality_ == ResampleQuality::High) {
        std::cerr << "[Resampler] High quality requested but libsamplerate not available, "
                  << "falling back to Medium quality" << std::endl;
        quality_ = ResampleQuality::Medium;
    }
#endif
}

void Resampler::process(AudioBuffer& buffer) {
    if (!isActive()) return;  // Passthrough
    
    switch (quality_) {
        case ResampleQuality::Simple:
            processSimple(buffer);
            break;
        case ResampleQuality::Medium:
            processMedium(buffer);
            break;
        case ResampleQuality::High:
            processHigh(buffer);
            break;
    }
}

void Resampler::processSimple(AudioBuffer& buffer) {
    // Linear interpolation resampling
    const float ratio = static_cast<float>(outputRate_) / inputRate_;
    const int inputFrames = buffer.getFrameCount();
    const int outputFrames = static_cast<int>(std::ceil(inputFrames * ratio));
    
    AudioBuffer outputBuffer(buffer.getChannelCount(), outputFrames);
    
    for (int ch = 0; ch < buffer.getChannelCount(); ++ch) {
        const float* input = buffer.getChannelData(ch);
        float* output = outputBuffer.getChannelData(ch);
        
        for (int i = 0; i < outputFrames; ++i) {
            float srcPos = i / ratio;
            int srcIndex = static_cast<int>(srcPos);
            float frac = srcPos - srcIndex;
            
            if (srcIndex + 1 < inputFrames) {
                // Linear interpolation between two samples
                output[i] = input[srcIndex] * (1.0f - frac) + 
                           input[srcIndex + 1] * frac;
            } else if (srcIndex < inputFrames) {
                // Last sample - interpolate with continuity buffer
                output[i] = input[srcIndex] * (1.0f - frac) +
                           lastSamples_[ch] * frac;
            } else {
                // Beyond input range - use continuity buffer
                output[i] = lastSamples_[ch];
            }
        }
        
        // Update continuity buffer for next iteration
        lastSamples_[ch] = input[inputFrames - 1];
    }
    
    buffer = std::move(outputBuffer);
}

void Resampler::processMedium(AudioBuffer& buffer) {
    // Cubic interpolation (Catmull-Rom spline) for better quality
    const float ratio = static_cast<float>(outputRate_) / inputRate_;
    const int inputFrames = buffer.getFrameCount();
    const int outputFrames = static_cast<int>(std::ceil(inputFrames * ratio));
    
    AudioBuffer outputBuffer(buffer.getChannelCount(), outputFrames);
    
    for (int ch = 0; ch < buffer.getChannelCount(); ++ch) {
        const float* input = buffer.getChannelData(ch);
        float* output = outputBuffer.getChannelData(ch);
        
        for (int i = 0; i < outputFrames; ++i) {
            float srcPos = i / ratio;
            int srcIndex = static_cast<int>(srcPos);
            float frac = srcPos - srcIndex;
            
            // Get 4-point neighborhood for cubic interpolation
            float p0 = (srcIndex > 0) ? input[srcIndex - 1] : lastSamples_[ch];
            float p1 = (srcIndex < inputFrames) ? input[srcIndex] : lastSamples_[ch];
            float p2 = (srcIndex + 1 < inputFrames) ? input[srcIndex + 1] : lastSamples_[ch];
            float p3 = (srcIndex + 2 < inputFrames) ? input[srcIndex + 2] : lastSamples_[ch];
            
            // Catmull-Rom spline coefficients
            float a = -0.5f*p0 + 1.5f*p1 - 1.5f*p2 + 0.5f*p3;
            float b = p0 - 2.5f*p1 + 2.0f*p2 - 0.5f*p3;
            float c = -0.5f*p0 + 0.5f*p2;
            float d = p1;
            
            // Evaluate cubic polynomial
            output[i] = a*frac*frac*frac + b*frac*frac + c*frac + d;
        }
        
        // Update continuity buffer
        lastSamples_[ch] = input[inputFrames - 1];
    }
    
    buffer = std::move(outputBuffer);
}

void Resampler::processHigh(AudioBuffer& buffer) {
#ifdef HAVE_LIBSAMPLERATE
    if (!srcState_) {
        // Fallback to medium if libsamplerate not initialized
        processMedium(buffer);
        return;
    }
    
    const int inputFrames = buffer.getFrameCount();
    const double ratio = static_cast<double>(outputRate_) / inputRate_;
    const int outputFrames = static_cast<int>(std::ceil(inputFrames * ratio));
    
    AudioBuffer outputBuffer(buffer.getChannelCount(), outputFrames);
    
    // libsamplerate requires interleaved data
    std::vector<float> interleavedInput(inputFrames * channels_);
    std::vector<float> interleavedOutput(outputFrames * channels_);
    
    // Interleave input channels
    for (int f = 0; f < inputFrames; ++f) {
        for (int ch = 0; ch < channels_; ++ch) {
            interleavedInput[f * channels_ + ch] = buffer.getChannelData(ch)[f];
        }
    }
    
    // Configure libsamplerate
    SRC_DATA srcData;
    srcData.data_in = interleavedInput.data();
    srcData.data_out = interleavedOutput.data();
    srcData.input_frames = inputFrames;
    srcData.output_frames = outputFrames;
    srcData.src_ratio = ratio;
    srcData.end_of_input = 0;  // Continuous stream
    
    // Process
    int error = src_process(static_cast<SRC_STATE*>(srcState_), &srcData);
    if (error) {
        std::cerr << "[Resampler] libsamplerate error: " << src_strerror(error)
                  << ", falling back to medium quality" << std::endl;
        processMedium(buffer);
        return;
    }
    
    // De-interleave output
    for (int f = 0; f < srcData.output_frames_gen; ++f) {
        for (int ch = 0; ch < channels_; ++ch) {
            outputBuffer.getChannelData(ch)[f] = interleavedOutput[f * channels_ + ch];
        }
    }
    
    // Adjust buffer size if different from expected
    if (srcData.output_frames_gen != outputFrames) {
        outputBuffer.resize(channels_, srcData.output_frames_gen);
    }
    
    buffer = std::move(outputBuffer);
    
    // Update continuity buffer
    for (int ch = 0; ch < channels_; ++ch) {
        lastSamples_[ch] = buffer.getChannelData(ch)[buffer.getFrameCount() - 1];
    }
#else
    // Fallback to medium if libsamplerate not available
    processMedium(buffer);
#endif
}

} // namespace nda

