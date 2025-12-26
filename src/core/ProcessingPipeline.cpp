#include "core/ProcessingPipeline.h"
#include <thread>
#include <chrono>
#include <cstring>
#include <iostream>
#include <algorithm>
#include <fstream>
#include <cmath>

namespace nda {

ProcessingPipeline::ProcessingPipeline()
    : isRunning_(false)
    , processingThread_(nullptr)
    , frameCount_(512)
    , processedSamples_(0)
    , targetSampleRate_(48000)  // Default: 48kHz internal processing
    , droppedSamples_(0)
    , driftWarnings_(0)
    , backpressureWaits_(0)
    , consecutiveFailures_(0)
    , processorFailures_(0)
    , peakLeft_(0.0f)
    , peakRight_(0.0f)
{
}

ProcessingPipeline::~ProcessingPipeline()
{
    if (isRunning_) {
        stop();
    }
    shutdown();
}

bool ProcessingPipeline::setSource(std::shared_ptr<AudioSourcePlugin> source)
{
    if (isRunning_) return false;
    source_ = source;
    return true;
}

bool ProcessingPipeline::setProcessor(std::shared_ptr<AudioProcessorPlugin> processor)
{
    if (isRunning_) return false;
    processor_ = processor;  // Optional - can be nullptr
    return true;
}

bool ProcessingPipeline::setSink(std::shared_ptr<AudioSinkPlugin> sink)
{
    if (isRunning_) return false;
    sink_ = sink;
    return true;
}

bool ProcessingPipeline::initialize()
{
    // Validate required plugins
    if (!source_ || !sink_) {
        std::cerr << "[Pipeline] Error: Source and sink are required" << std::endl;
        return false;
    }
    
    // Initialize source
    auto sourceState = source_->getState();
    std::cout << "[Pipeline] Source: " << source_->getInfo().name
              << " (state: " << static_cast<int>(sourceState) << ")" << std::endl;

    if (sourceState == PluginState::Unloaded) {
        if (!source_->initialize()) {
            std::cerr << "[Pipeline] Source initialization failed" << std::endl;
            return false;
        }
    } else if (sourceState != PluginState::Initialized) {
        std::cerr << "[Pipeline] Source in invalid state for pipeline" << std::endl;
        return false;
    }

    // Initialize processor (optional)
    if (processor_) {
        auto processorState = processor_->getState();
        std::cout << "[Pipeline] Processor: " << processor_->getInfo().name
                  << " (state: " << static_cast<int>(processorState) << ")" << std::endl;

        if (processorState == PluginState::Unloaded) {
            if (!processor_->initialize()) {
                std::cerr << "[Pipeline] Processor initialization failed" << std::endl;
                return false;
            }
        } else if (processorState != PluginState::Initialized) {
            std::cerr << "[Pipeline] Processor in invalid state for pipeline" << std::endl;
            return false;
        }
    }

    // Initialize sink
    auto sinkState = sink_->getState();
    std::cout << "[Pipeline] Sink: " << sink_->getInfo().name
              << " (state: " << static_cast<int>(sinkState) << ")" << std::endl;

    if (sinkState == PluginState::Unloaded) {
        if (!sink_->initialize()) {
            std::cerr << "[Pipeline] Sink initialization failed" << std::endl;
            return false;
        }
    } else if (sinkState != PluginState::Initialized) {
        std::cerr << "[Pipeline] Sink in invalid state for pipeline" << std::endl;
        return false;
    }

    // Initialize work buffer
    int sourceRate = source_->getSampleRate();
    int sinkRate = sink_->getSampleRate();
    int channels = source_->getChannels();

    int desiredFrames = frameCount_;
    desiredFrames = std::min(desiredFrames, source_->getBufferSize());
    desiredFrames = std::min(desiredFrames, sink_->getBufferSize());
    
    if (desiredFrames <= 0) {
        desiredFrames = 512;
    }

    frameCount_ = desiredFrames;

    source_->setBufferSize(frameCount_);
    sink_->setBufferSize(frameCount_);
    
    // Configure processor for pipeline's internal sample rate
    if (processor_) {
        processor_->setSampleRate(targetSampleRate_);
        processor_->setChannelCount(channels);
    }

    workBuffer_.resize(channels, frameCount_);

    // Auto-configure resamplers if sample rates mismatch (v2.0 auto-fix)
    if (sourceRate != targetSampleRate_) {
        std::cout << "[Pipeline] Auto-resampling enabled: " << sourceRate 
                  << "Hz → " << targetSampleRate_ << "Hz (source)" << std::endl;
        sourceResampler_.initialize(sourceRate, targetSampleRate_, channels, ResampleQuality::Simple);
    }
    
    if (sinkRate != targetSampleRate_) {
        std::cout << "[Pipeline] Auto-resampling enabled: " << targetSampleRate_ 
                  << "Hz → " << sinkRate << "Hz (sink)" << std::endl;
        sinkResampler_.initialize(targetSampleRate_, sinkRate, channels, ResampleQuality::Simple);
    }

    std::cout << "[Pipeline] Initialization complete - " << channels << " channels @ "
              << targetSampleRate_ << "Hz internal (frame size " << frameCount_ << ")" << std::endl;
    return true;
}

bool ProcessingPipeline::start()
{
    if (isRunning_) return false;
    if (!source_ || !sink_) return false;

    // Start all plugins (3-slot model)
    if (!source_->start()) {
        std::cerr << "[Pipeline] Source start failed" << std::endl;
        return false;
    }
    
    if (processor_ && !processor_->start()) {
        std::cerr << "[Pipeline] Processor start failed" << std::endl;
        source_->stop();  // Rollback
        return false;
    }
    
    if (!sink_->start()) {
        std::cerr << "[Pipeline] Sink start failed" << std::endl;
        if (processor_) processor_->stop();
        source_->stop();  // Rollback
        return false;
    }

    // Reset metrics (v2.0)
    isRunning_ = true;
    processedSamples_ = 0;
    droppedSamples_ = 0;
    driftWarnings_ = 0;
    backpressureWaits_ = 0;
    consecutiveFailures_ = 0;
    processorFailures_ = 0;
    startTime_ = std::chrono::steady_clock::now();

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

    // CRITICAL FIX: Stop sink FIRST to end audio callbacks, preventing deadlock
    if (sink_) {
        sink_->stop();
    }
    
    // Stop source to unblock any pending read() calls
    if (source_) {
        source_->stop();
    }
    
    // Stop processor
    if (processor_) {
        processor_->stop();
    }

    // Now the processing thread can exit cleanly
    if (processingThread_ && processingThread_->joinable()) {
        processingThread_->join();
        processingThread_.reset();
    }

    double seconds = (double)processedSamples_ / (source_ ? source_->getSampleRate() : 48000);
    std::cout << "[Pipeline] Stopped after processing " << processedSamples_
             << " samples (" << seconds << " seconds)" << std::endl;
}

void ProcessingPipeline::shutdown()
{
    if (source_) source_->shutdown();
    if (processor_) processor_->shutdown();
    if (sink_) sink_->shutdown();
}

double ProcessingPipeline::getLatency() const
{
    double latency = 0.0;

    // Source buffer time
    if (source_) {
        int bufferSize = workBuffer_.getFrameCount();
        int sampleRate = source_->getSampleRate();
        latency += (double)bufferSize / sampleRate * 1000.0; // ms
    }

    // Processor latency (if declared)
    if (processor_) {
        latency += processor_->getProcessingLatency() * 1000.0; // convert seconds to ms
    }

    // Sink buffer time
    if (sink_) {
        int bufferSize = sink_->getBufferSize();
        int sampleRate = sink_->getSampleRate();
        latency += (double)bufferSize / sampleRate * 1000.0; // ms
    }

    return latency;
}

float ProcessingPipeline::getCPULoad() const
{
    // Legacy method - v1.x compatibility
    return getActualCPULoad();
}

// v2.0: Measured metrics implementations
float ProcessingPipeline::getActualCPULoad() const
{
    if (!isRunning_) return 0.0f;
    
    auto now = std::chrono::steady_clock::now();
    auto wallTime = std::chrono::duration_cast<std::chrono::milliseconds>(
        now - startTime_
    ).count();
    
    // Calculate audio time processed
    auto audioTime = (processedSamples_ * 1000) / targetSampleRate_;
    
    if (wallTime == 0) return 0.0f;
    
    // CPU% approximation: (audio time / wall time) * 100
    // Note: This is pipeline thread efficiency, not system CPU
    return (static_cast<float>(audioTime) / wallTime) * 100.0f;
}

double ProcessingPipeline::getActualLatency() const
{
    double latency = 0.0;
    
    // Source buffer time
    if (source_) {
        latency += (double)workBuffer_.getFrameCount() / targetSampleRate_;
    }
    
    // Processor latency (if declared)
    if (processor_ && processor_->getState() == PluginState::Running) {
        latency += processor_->getProcessingLatency();
    }
    
    // Sink buffer time
    if (sink_) {
        int sinkBufferSize = sink_->getBufferSize();
        int sinkRate = sink_->getSampleRate();
        latency += (double)sinkBufferSize / sinkRate;
    }
    
    return latency;  // Returns seconds
}

uint64_t ProcessingPipeline::getProcessedSamples() const
{
    return processedSamples_;
}

void ProcessingPipeline::getPeakLevels(float& left, float& right) const
{
    left = peakLeft_.load(std::memory_order_relaxed);
    right = peakRight_.load(std::memory_order_relaxed);
}

// v2.0: Runtime metrics
double ProcessingPipeline::getUptime() const
{
    if (!isRunning_) return 0.0;
    
    auto now = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        now - startTime_
    );
    
    return duration.count() / 1000.0;  // Convert to seconds
}

double ProcessingPipeline::getRealTimeRatio() const
{
    if (!isRunning_ || processedSamples_ == 0) return 0.0;
    
    auto now = std::chrono::steady_clock::now();
    auto wallTime = std::chrono::duration_cast<std::chrono::microseconds>(
        now - startTime_
    ).count();
    
    if (wallTime == 0) return 0.0;
    
    // Calculate audio time processed (in microseconds)
    auto audioTime = (processedSamples_ * 1000000) / targetSampleRate_;
    
    // Real-time ratio: audio time / wall time
    // 1.0 = perfect real-time, <1.0 = slower than real-time, >1.0 = faster
    return static_cast<double>(audioTime) / wallTime;
}

void ProcessingPipeline::processingThread()
{
    std::cout << "[Pipeline] Processing thread started with real-time pacing" << std::endl;

    int frameCount = 0;

    while (isRunning_) {
        processAudioFrame();
        frameCount++;

        // v2.0: Real-time pacing - sleep to maintain exact 1.0x real-time cadence
        auto targetTime = startTime_ + std::chrono::microseconds(
            (processedSamples_ * 1000000) / targetSampleRate_
        );
        
        auto now = std::chrono::steady_clock::now();
        
        if (now < targetTime) {
            // We're ahead of schedule - sleep until target time
            std::this_thread::sleep_until(targetTime);
        } else {
            // We're behind schedule - track drift
            auto drift = std::chrono::duration_cast<std::chrono::milliseconds>(now - targetTime);
            if (drift.count() > 50) {
                // More than 50ms behind - log warning
                driftWarnings_++;
                if (driftWarnings_ % 100 == 0) {
                    std::cerr << "[Pipeline] WARNING: " << drift.count() 
                              << "ms behind schedule (drift #" << driftWarnings_ << ")" << std::endl;
                }
            }
        }

        // Log progress every second
        if (frameCount % 100 == 0) {  // ~100 frames = ~1 second at 512 samples/frame
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - startTime_).count();
            double audioSeconds = (double)processedSamples_ / targetSampleRate_;
            std::cout << "[Pipeline] Running for " << elapsed << "s, processed "
                     << processedSamples_ << " samples (" << audioSeconds << "s of audio)";
            if (droppedSamples_ > 0) {
                std::cout << ", dropped: " << droppedSamples_;
            }
            std::cout << std::endl;
        }
    }

    std::cout << "[Pipeline] Processing thread exiting" << std::endl;
}

void ProcessingPipeline::processAudioFrame()
{
    if (!source_ || !sink_) return;

    // 1. Read audio from source
    if (!source_->readAudio(workBuffer_)) {
        consecutiveFailures_++;

        if (consecutiveFailures_ == 1) {
            std::cerr << "[Pipeline] Audio read started failing" << std::endl;
        }

        if (consecutiveFailures_ > 10 && consecutiveFailures_ % 100 == 0) {
            std::cerr << "[Pipeline] " << consecutiveFailures_ << " consecutive read failures" << std::endl;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        return;
    }

    // Reset failure counter on success
    if (consecutiveFailures_ > 0) {
        std::cerr << "[Pipeline] Audio read recovered after " << consecutiveFailures_ << " failures" << std::endl;
        consecutiveFailures_ = 0;
    }

    // 2. Resample source → 48kHz (auto-applied if source rate differs)
    if (sourceResampler_.isActive()) {
        sourceResampler_.process(workBuffer_);
    }

    // 3. Apply processor at 48kHz (optional - encryptor, decryptor, effects, etc.)
    if (processor_ && processor_->getState() == PluginState::Running) {
        if (!processor_->processAudio(workBuffer_)) {
            // Processor failed - log but continue with passthrough
            processorFailures_++;
            if (processorFailures_ == 1 || processorFailures_ % 100 == 0) {
                std::cerr << "[Pipeline] Processor failed (" << processorFailures_ 
                         << " total failures), passing through unprocessed audio" << std::endl;
            }
        }
    }

    // 4. Resample 48kHz → sink rate (auto-applied if sink rate differs)
    if (sinkResampler_.isActive()) {
        sinkResampler_.process(workBuffer_);
    }

    int channels = workBuffer_.getChannelCount();
    int frames = workBuffer_.getFrameCount();
    if (channels > 0 && frames > 0) {
        const float* left = workBuffer_.getChannelData(0);
        const float* right = (channels > 1) ? workBuffer_.getChannelData(1) : left;
        float leftPeak = 0.0f;
        float rightPeak = 0.0f;

        for (int i = 0; i < frames; ++i) {
            leftPeak = std::max(leftPeak, std::abs(left[i]));
            rightPeak = std::max(rightPeak, std::abs(right[i]));
        }

        if (leftPeak > 1.0f) leftPeak = 1.0f;
        if (rightPeak > 1.0f) rightPeak = 1.0f;

        peakLeft_.store(leftPeak, std::memory_order_relaxed);
        peakRight_.store(rightPeak, std::memory_order_relaxed);
    } else {
        peakLeft_.store(0.0f, std::memory_order_relaxed);
        peakRight_.store(0.0f, std::memory_order_relaxed);
    }

    // 5. Check backpressure (v2.0) - wait if sink queue is full
    int available = sink_->getAvailableSpace();
    if (available < workBuffer_.getFrameCount()) {
        // Sink queue is full - wait briefly for space
        backpressureWaits_++;
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        
        // Retry once
        available = sink_->getAvailableSpace();
        if (available < workBuffer_.getFrameCount()) {
            // Still no space - drop this buffer
            droppedSamples_ += workBuffer_.getFrameCount();
            if (droppedSamples_ <= 100) {  // Log first 100 drops
                std::cerr << "[Pipeline] Buffer dropped (sink queue full)" << std::endl;
            }
            return;
        }
    }

    // 6. Write to sink (only increment processedSamples on success)
    if (sink_->writeAudio(workBuffer_)) {
        processedSamples_ += workBuffer_.getFrameCount();
    } else {
        droppedSamples_ += workBuffer_.getFrameCount();
        std::cerr << "[Pipeline] Sink write failed" << std::endl;
    }
}

} // namespace nda
