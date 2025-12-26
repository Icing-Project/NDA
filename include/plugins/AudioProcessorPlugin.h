#ifndef AUDIOPROCESSORPLUGIN_H
#define AUDIOPROCESSORPLUGIN_H

#include "BasePlugin.h"
#include "audio/AudioBuffer.h"
#include <memory>

namespace nda {

/**
 * @brief Audio processor plugin interface
 * 
 * Processes audio in-place (encryption, decryption, effects, resampling, etc.)
 * 
 * @thread-safety processAudio() is called from the pipeline thread only.
 * Multiple processors must not share state unless explicitly synchronized.
 * 
 * @note Processors operate at the pipeline's internal sample rate (typically 48kHz).
 * The pipeline handles resampling before/after processor invocation.
 * 
 * @example
 * class GainProcessor : public AudioProcessorPlugin {
 *     bool processAudio(AudioBuffer& buffer) override {
 *         for (int ch = 0; ch < buffer.getChannelCount(); ++ch) {
 *             for (int f = 0; f < buffer.getFrameCount(); ++f) {
 *                 buffer.getChannelData(ch)[f] *= gain_;
 *             }
 *         }
 *         return true;
 *     }
 * };
 */
class AudioProcessorPlugin : public virtual BasePlugin {
public:
    virtual ~AudioProcessorPlugin() = default;
    
    /**
     * @brief Process audio buffer in-place
     * 
     * @param buffer Audio data to process (modified in-place)
     * @return true if processing succeeded, false on error
     * 
     * @note On false return, pipeline will passthrough buffer unchanged.
     * Processor should handle errors gracefully and log appropriately.
     */
    virtual bool processAudio(AudioBuffer& buffer) = 0;
    
    /**
     * @brief Get current sample rate
     * @return Sample rate in Hz (typically 48000)
     */
    virtual int getSampleRate() const = 0;
    
    /**
     * @brief Get current channel count
     * @return Number of channels (typically 2 for stereo)
     */
    virtual int getChannelCount() const = 0;
    
    /**
     * @brief Set sample rate (called during pipeline initialization)
     * @param rate Sample rate in Hz
     */
    virtual void setSampleRate(int rate) = 0;
    
    /**
     * @brief Set channel count (called during pipeline initialization)
     * @param channels Number of channels
     */
    virtual void setChannelCount(int channels) = 0;
    
    /**
     * @brief Get processing latency added by this processor
     * @return Latency in seconds (default: 0.0 for zero-latency processors)
     * 
     * @note Used for latency reporting in dashboard.
     * Includes algorithmic latency (e.g., lookahead buffers), not computational time.
     */
    virtual double getProcessingLatency() const { return 0.0; }
};

} // namespace nda

#endif // AUDIOPROCESSORPLUGIN_H

