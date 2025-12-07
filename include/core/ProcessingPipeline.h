#ifndef PROCESSINGPIPELINE_H
#define PROCESSINGPIPELINE_H

#include "plugins/AudioSourcePlugin.h"
#include "plugins/BearerPlugin.h"
#include "plugins/EncryptorPlugin.h"
#include "plugins/AudioSinkPlugin.h"
#include "audio/AudioBuffer.h"
#include <memory>
#include <vector>
#include <thread>
#include <atomic>
#include <deque>
#include <mutex>

namespace nda {

class ProcessingPipeline {
public:
    ProcessingPipeline();
    ~ProcessingPipeline();

    // Pipeline configuration
    bool setAudioSource(std::shared_ptr<AudioSourcePlugin> source);
    bool setBearer(std::shared_ptr<BearerPlugin> bearer);
    bool setEncryptor(std::shared_ptr<EncryptorPlugin> encryptor);
    bool setAudioSink(std::shared_ptr<AudioSinkPlugin> sink);

    // Pipeline control
    bool initialize();
    bool start();
    void stop();
    void shutdown();

    // State
    bool isRunning() const { return isRunning_; }

    // Statistics
    double getLatency() const;
    float getCPULoad() const;
    uint64_t getProcessedSamples() const;

private:
    void processingThread();
    void processAudioFrame();
    bool drainIncomingAudio();
    void handleReceivedPacket(const Packet& packet);

    std::vector<uint8_t> serializeAudioBuffer(const AudioBuffer& buffer) const;
    bool deserializeAudioBuffer(const std::vector<uint8_t>& data, int channels, AudioBuffer& buffer) const;

    std::shared_ptr<AudioSourcePlugin> audioSource_;
    std::shared_ptr<BearerPlugin> bearer_;
    std::shared_ptr<EncryptorPlugin> encryptor_;
    std::shared_ptr<AudioSinkPlugin> audioSink_;

    std::atomic<bool> isRunning_;
    std::unique_ptr<std::thread> processingThread_;

    AudioBuffer workBuffer_;
    uint64_t processedSamples_;

    std::deque<AudioBuffer> incomingAudio_;
    mutable std::mutex incomingMutex_;
};

} // namespace nda

#endif // PROCESSINGPIPELINE_H
