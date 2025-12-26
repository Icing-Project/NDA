#ifndef RESAMPLER_H
#define RESAMPLER_H

#include "AudioBuffer.h"
#include <vector>

namespace nda {

/**
 * @brief Sample rate conversion quality modes
 */
enum class ResampleQuality {
    Simple,   // Linear interpolation (fast, -60dB THD+N)
    Medium,   // Cubic interpolation (balanced, -80dB THD+N)
    High      // libsamplerate (slow, -120dB THD+N)
};

/**
 * @brief Audio sample rate converter
 * 
 * Handles automatic sample rate conversion for pipeline compatibility.
 * Uses different algorithms based on quality settings.
 * 
 * @thread-safety Not thread-safe. Call from single pipeline thread only.
 * 
 * @note Maintains continuity between buffers to avoid clicks/pops.
 */
class Resampler {
public:
    Resampler();
    ~Resampler();  // Clean up libsamplerate state if used
    
    /**
     * @brief Initialize resampler
     * 
     * Auto-fix mode: if rates mismatch, auto-enable resampling.
     * 
     * @param inputRate Source sample rate in Hz
     * @param outputRate Target sample rate in Hz
     * @param channels Number of channels
     * @param quality Resampling quality (default: Simple)
     */
    void initialize(int inputRate, int outputRate, int channels,
                   ResampleQuality quality = ResampleQuality::Simple);
    
    /**
     * @brief Process buffer (resample in-place or to new buffer)
     * 
     * @param buffer Input/output buffer (may be resized)
     * 
     * @note If input/output rates match, this is a no-op (zero overhead).
     * Buffer size will change based on rate ratio.
     */
    void process(AudioBuffer& buffer);
    
    /**
     * @brief Check if resampling is active
     * @return true if rates differ, false if passthrough
     */
    bool isActive() const { return inputRate_ != outputRate_; }
    
    int getInputRate() const { return inputRate_; }
    int getOutputRate() const { return outputRate_; }
    ResampleQuality getQuality() const { return quality_; }
    
private:
    int inputRate_;
    int outputRate_;
    int channels_;
    ResampleQuality quality_;
    
    // For continuity between buffers (avoid clicks/pops at buffer boundaries)
    std::vector<float> lastSamples_;
    
    // Quality mode implementations
    void processSimple(AudioBuffer& buffer);   // Linear interpolation
    void processMedium(AudioBuffer& buffer);   // Cubic (Catmull-Rom)
    void processHigh(AudioBuffer& buffer);     // libsamplerate
    
#ifdef HAVE_LIBSAMPLERATE
    void* srcState_;  // SRC_STATE* (opaque to avoid header dependency)
#endif
};

} // namespace nda

#endif // RESAMPLER_H

