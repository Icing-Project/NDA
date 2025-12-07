#include "core/ProcessingPipeline.h"
#include <thread>
#include <chrono>
#include <cstring>
#include <iostream>
#include <vector>
#include <algorithm>

namespace NADE {

ProcessingPipeline::ProcessingPipeline()
    : isRunning_(false), processingThread_(nullptr), processedSamples_(0)
{
}

ProcessingPipeline::~ProcessingPipeline()
{
    if (isRunning_) {
        stop();
    }
    shutdown();
}

bool ProcessingPipeline::setAudioSource(std::shared_ptr<AudioSourcePlugin> source)
{
    if (isRunning_) return false;
    audioSource_ = source;
    return true;
}

bool ProcessingPipeline::setBearer(std::shared_ptr<BearerPlugin> bearer)
{
    if (isRunning_) return false;
    bearer_ = bearer;
    return true;
}

bool ProcessingPipeline::setEncryptor(std::shared_ptr<EncryptorPlugin> encryptor)
{
    if (isRunning_) return false;
    encryptor_ = encryptor;
    return true;
}

bool ProcessingPipeline::setAudioSink(std::shared_ptr<AudioSinkPlugin> sink)
{
    if (isRunning_) return false;
    audioSink_ = sink;
    return true;
}

bool ProcessingPipeline::initialize()
{
    // Initialize all plugins in the pipeline (only if not already initialized)
    if (audioSource_) {
        auto state = audioSource_->getState();
        std::cout << "[Pipeline] Audio source: " << audioSource_->getInfo().name
                  << " (state: " << static_cast<int>(state) << ")" << std::endl;

        if (state == PluginState::Unloaded) {
            if (!audioSource_->initialize()) {
                std::cerr << "[Pipeline] Audio source initialization failed" << std::endl;
                return false;
            }
        } else if (state != PluginState::Initialized) {
            std::cerr << "[Pipeline] Audio source in invalid state for pipeline" << std::endl;
            return false;
        }
    }

    if (bearer_) {
        auto state = bearer_->getState();
        std::cout << "[Pipeline] Bearer: " << bearer_->getInfo().name
                  << " (state: " << static_cast<int>(state) << ")" << std::endl;

        if (state == PluginState::Unloaded) {
            if (!bearer_->initialize()) {
                std::cerr << "[Pipeline] Bearer initialization failed" << std::endl;
                return false;
            }
        } else if (state != PluginState::Initialized) {
            std::cerr << "[Pipeline] Bearer in invalid state for pipeline" << std::endl;
            return false;
        }
    }

    if (encryptor_) {
        auto state = encryptor_->getState();
        std::cout << "[Pipeline] Encryptor: " << encryptor_->getInfo().name
                  << " (state: " << static_cast<int>(state) << ")" << std::endl;

        if (state == PluginState::Unloaded) {
            if (!encryptor_->initialize()) {
                std::cerr << "[Pipeline] Encryptor initialization failed" << std::endl;
                return false;
            }
        } else if (state != PluginState::Initialized) {
            std::cerr << "[Pipeline] Encryptor in invalid state for pipeline" << std::endl;
            return false;
        }
    }

    if (audioSink_) {
        auto state = audioSink_->getState();
        std::cout << "[Pipeline] Audio sink: " << audioSink_->getInfo().name
                  << " (state: " << static_cast<int>(state) << ")" << std::endl;

        if (state == PluginState::Unloaded) {
            if (!audioSink_->initialize()) {
                std::cerr << "[Pipeline] Audio sink initialization failed" << std::endl;
                return false;
            }
        } else if (state != PluginState::Initialized) {
            std::cerr << "[Pipeline] Audio sink in invalid state for pipeline" << std::endl;
            return false;
        }
    }

    // Initialize work buffer
    int sampleRate = audioSource_ ? audioSource_->getSampleRate() : 48000;
    int channels = audioSource_ ? audioSource_->getChannels() : 2;
    workBuffer_.resize(channels, 512);

    std::cout << "[Pipeline] Initialization complete - " << channels << " channels @ " << sampleRate << "Hz" << std::endl;
    return true;
}

bool ProcessingPipeline::start()
{
    if (isRunning_) return false;
    if (!audioSource_ || !audioSink_) return false;

    // Wire bearer callback for duplex receive path before threads start
    if (bearer_) {
        bearer_->setPacketReceivedCallback([this](const Packet& packet) {
            this->handleReceivedPacket(packet);
        });
    }

    // Start all plugins
    if (!audioSink_->start()) return false;
    if (bearer_ && !bearer_->start()) return false;
    if (encryptor_ && !encryptor_->start()) return false;
    if (!audioSource_->start()) return false;

    isRunning_ = true;
    processedSamples_ = 0;  // Reset counter

    std::cout << "[Pipeline] Starting processing thread..." << std::endl;

    // Start processing thread (keep it joinable, don't detach)
    processingThread_ = std::make_unique<std::thread>([this]() {
        this->processingThread();
    });

    return true;
}

void ProcessingPipeline::stop()
{
    if (!isRunning_) return;

    std::cout << "[Pipeline] Stopping... (processed " << processedSamples_ << " samples so far)" << std::endl;

    // Signal thread to stop
    isRunning_ = false;

    // CRITICAL: Stop audio source first to unblock any pending read() calls
    // Otherwise the processing thread will hang in PyAudio's blocking read
    if (audioSource_) audioSource_->stop();

    // Now the processing thread can exit cleanly
    if (processingThread_ && processingThread_->joinable()) {
        processingThread_->join();
        processingThread_.reset();
    }

    // Stop remaining plugins
    if (bearer_) bearer_->stop();
    if (encryptor_) encryptor_->stop();
    if (audioSink_) audioSink_->stop();

    double seconds = (double)processedSamples_ / (audioSource_ ? audioSource_->getSampleRate() : 48000);
    std::cout << "[Pipeline] Stopped after processing " << processedSamples_
             << " samples (" << seconds << " seconds)" << std::endl;
}

void ProcessingPipeline::shutdown()
{
    if (audioSource_) audioSource_->shutdown();
    if (bearer_) bearer_->shutdown();
    if (encryptor_) encryptor_->shutdown();
    if (audioSink_) audioSink_->shutdown();
}

double ProcessingPipeline::getLatency() const
{
    double latency = 0.0;

    // Add latency from each component
    if (audioSource_) {
        int bufferSize = workBuffer_.getFrameCount();
        int sampleRate = audioSource_->getSampleRate();
        latency += (double)bufferSize / sampleRate * 1000.0; // ms
    }

    if (bearer_) {
        latency += bearer_->getLatency();
    }

    if (audioSink_) {
        int bufferSize = audioSink_->getBufferSize();
        int sampleRate = audioSink_->getSampleRate();
        latency += (double)bufferSize / sampleRate * 1000.0; // ms
    }

    return latency;
}

float ProcessingPipeline::getCPULoad() const
{
    // Placeholder - would measure actual CPU time
    return 5.0f;
}

uint64_t ProcessingPipeline::getProcessedSamples() const
{
    return processedSamples_;
}

void ProcessingPipeline::processingThread()
{
    std::cout << "[Pipeline] Processing thread started" << std::endl;

    auto startTime = std::chrono::steady_clock::now();
    int frameCount = 0;
    int sampleRate = audioSource_ ? audioSource_->getSampleRate() : 48000;

    while (isRunning_) {
        while (drainIncomingAudio()) {
            // Drain queued inbound audio before handling new capture frames
        }

        processAudioFrame();
        frameCount++;

        // Log progress every second
        if (frameCount % 100 == 0) {  // ~100 frames = ~1 second at 512 samples/frame
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - startTime).count();
            double audioSeconds = (double)processedSamples_ / sampleRate;
            std::cout << "[Pipeline] Running for " << elapsed << "s, processed "
                     << processedSamples_ << " samples (" << audioSeconds << "s of audio)" << std::endl;
        }

        // NO SLEEP - readAudio() blocks with timeout, naturally pacing the loop
    }

    std::cout << "[Pipeline] Processing thread exiting" << std::endl;
}

void ProcessingPipeline::processAudioFrame()
{
    if (!audioSource_ || !audioSink_) return;

    static int consecutiveFailures = 0;

    // 1. Read audio from source
    if (!audioSource_->readAudio(workBuffer_)) {
        consecutiveFailures++;

        if (consecutiveFailures == 1) {
            std::cerr << "[Pipeline] Audio read started failing" << std::endl;
        }

        if (consecutiveFailures > 10 && consecutiveFailures % 100 == 0) {
            std::cerr << "[Pipeline] " << consecutiveFailures << " consecutive read failures" << std::endl;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        return;
    }

    // Reset failure counter on success
    if (consecutiveFailures > 0) {
        std::cerr << "[Pipeline] Audio read recovered after " << consecutiveFailures << " failures" << std::endl;
        consecutiveFailures = 0;
    }

    // 2. Encrypt payload for transport if encryptor is available
    std::vector<uint8_t> payload = serializeAudioBuffer(workBuffer_);
    if (encryptor_) {
        std::vector<uint8_t> plaintext = payload;

        // Generate nonce (would be more sophisticated in real implementation)
        uint8_t nonce[12] = {0};

        const size_t tagSize = static_cast<size_t>(std::max(0, encryptor_->getTagSize()));
        size_t outputSize = payload.size() + tagSize;
        payload.resize(outputSize);

        if (!encryptor_->encrypt(plaintext.data(), plaintext.size(), payload.data(), outputSize, nonce, 12)) {
            std::cerr << "[Pipeline] Failed to encrypt payload, dropping frame" << std::endl;
            return;
        }

        payload.resize(outputSize);
    }

    // 3. Send over bearer if available
    if (bearer_ && bearer_->isConnected()) {
        Packet packet;

        // Serialize audio buffer to packet
        packet.data = payload;
        packet.timestamp = processedSamples_;
        packet.sequenceNumber = processedSamples_ / workBuffer_.getFrameCount();
        packet.flags = 0;

        bearer_->sendPacket(packet);
    }

    // 4. Write to audio sink
    audioSink_->writeAudio(workBuffer_);

    processedSamples_ += workBuffer_.getFrameCount();
}

bool ProcessingPipeline::drainIncomingAudio()
{
    AudioBuffer buffer;

    {
        std::lock_guard<std::mutex> lock(incomingMutex_);
        if (incomingAudio_.empty()) {
            return false;
        }

        buffer = incomingAudio_.front();
        incomingAudio_.pop_front();
    }

    audioSink_->writeAudio(buffer);
    processedSamples_ += buffer.getFrameCount();
    return true;
}

void ProcessingPipeline::handleReceivedPacket(const Packet& packet)
{
    if (!isRunning_ || !audioSink_) return;

    std::vector<uint8_t> payload(packet.data.begin(), packet.data.end());

    if (encryptor_) {
        std::vector<uint8_t> decrypted(payload.size());
        size_t outputSize = decrypted.size();
        uint8_t nonce[12] = {0};

        if (!encryptor_->decrypt(payload.data(), payload.size(), decrypted.data(), outputSize, nonce, 12)) {
            std::cerr << "[Pipeline] Failed to decrypt incoming packet" << std::endl;
            return;
        }

        decrypted.resize(outputSize);
        payload.swap(decrypted);
    }

    AudioBuffer buffer(audioSink_->getChannels(), 512);
    if (!deserializeAudioBuffer(payload, audioSink_->getChannels(), buffer)) {
        std::cerr << "[Pipeline] Dropping packet: unable to decode audio payload" << std::endl;
        return;
    }

    {
        std::lock_guard<std::mutex> lock(incomingMutex_);
        if (incomingAudio_.size() > 32) {
            incomingAudio_.pop_front();
        }
        incomingAudio_.push_back(buffer);
    }
}

std::vector<uint8_t> ProcessingPipeline::serializeAudioBuffer(const AudioBuffer& buffer) const
{
    int channels = buffer.getChannelCount();
    int frames = buffer.getFrameCount();

    std::vector<float> interleaved(frames * channels, 0.0f);
    for (int frame = 0; frame < frames; ++frame) {
        for (int ch = 0; ch < channels; ++ch) {
            const float* channelData = buffer.getChannelData(ch);
            interleaved[frame * channels + ch] = channelData ? channelData[frame] : 0.0f;
        }
    }

    const uint8_t* byteData = reinterpret_cast<const uint8_t*>(interleaved.data());
    return std::vector<uint8_t>(byteData, byteData + interleaved.size() * sizeof(float));
}

bool ProcessingPipeline::deserializeAudioBuffer(const std::vector<uint8_t>& data, int channels, AudioBuffer& buffer) const
{
    if (channels <= 0) return false;
    if (data.size() < sizeof(float) * static_cast<size_t>(channels)) return false;

    size_t frameCount = data.size() / (sizeof(float) * channels);
    const float* interleaved = reinterpret_cast<const float*>(data.data());

    buffer.resize(channels, static_cast<int>(frameCount));

    for (size_t frame = 0; frame < frameCount; ++frame) {
        for (int ch = 0; ch < channels; ++ch) {
            float* channelData = buffer.getChannelData(ch);
            if (channelData) {
                channelData[frame] = interleaved[frame * channels + ch];
            }
        }
    }

    return true;
}

} // namespace NADE
