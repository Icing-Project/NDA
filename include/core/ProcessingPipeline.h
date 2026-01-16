#ifndef PROCESSINGPIPELINE_H
#define PROCESSINGPIPELINE_H

#include "plugins/AudioSourcePlugin.h"
#include "plugins/AudioProcessorPlugin.h"
#include "plugins/AudioSinkPlugin.h"
#include "audio/AudioBuffer.h"
#include "audio/Resampler.h"
#include <memory>
#include <vector>
#include <thread>
#include <atomic>
#include <chrono>
#include <mutex>
#include <condition_variable>
#include <functional>

namespace nda {

class ProcessingPipeline {
public:
    ProcessingPipeline();
    ~ProcessingPipeline();

    // Pipeline configuration (3-slot architecture)
    bool setSource(std::shared_ptr<AudioSourcePlugin> source);
    bool setProcessor(std::shared_ptr<AudioProcessorPlugin> processor);  // Optional
    bool setSink(std::shared_ptr<AudioSinkPlugin> sink);

    // Pipeline control
    bool initialize();
    bool start();
    void stop();
    void shutdown();

    // Bridge Mode preset (v2.1 stabilization)
    void enableBridgeMode();

    // State
    bool isRunning() const { return isRunning_; }

    // Statistics
    double getLatency() const;
    float getCPULoad() const;
    uint64_t getProcessedSamples() const;
    void getPeakLevels(float& left, float& right) const;
    
    // v2.0: Measured metrics (not hardcoded)
    uint64_t getDroppedSamples() const { return droppedSamples_; }
    double getActualLatency() const;
    float getActualCPULoad() const;
    
    // v2.0: Diagnostic counters
    uint64_t getDriftWarnings() const { return driftWarnings_; }
    uint64_t getBackpressureWaits() const { return backpressureWaits_; }
    uint64_t getConsecutiveFailures() const { return consecutiveFailures_; }
    uint64_t getProcessorFailures() const { return processorFailures_; }

    // v2.1: Drift and timing metrics
    double getCurrentDriftMs() const;     // Current drift from real-time target
    double getMaxDriftMs() const;         // Maximum drift seen this session

    // v2.1: Underrun/overrun counters
    uint64_t getReadFailures() const;     // Source read failures (underruns)
    uint64_t getWriteFailures() const;    // Sink write failures (overruns)

    // v2.1: Health indicator
    enum class HealthStatus {
        OK,         // All metrics nominal
        Degraded,   // Drift >10ms or failures >10
        Failing     // Drift >50ms or failures >100
    };
    HealthStatus getHealthStatus() const;

    // v2.0: Runtime metrics
    double getUptime() const;
    double getRealTimeRatio() const;

    // ===== v2.2: Event-Driven Pipeline Support =====

    /**
     * @brief Pipeline scheduling mode
     */
    enum class SchedulingMode {
        Polling,      // Legacy: sleep_until() based timing
        EventDriven   // v2.2: Condition variable wake-ups from plugins
    };

    /**
     * @brief Get current scheduling mode
     */
    SchedulingMode getSchedulingMode() const { return schedulingMode_; }

    /**
     * @brief Check if pipeline is using event-driven scheduling
     */
    bool isEventDriven() const { return schedulingMode_ == SchedulingMode::EventDriven; }

    /**
     * @brief Callback invoked by source plugins when data is ready.
     *
     * Thread-safe. Wakes the pipeline thread via condition variable.
     * Called from source plugin background threads.
     */
    void onSourceDataReady();

    /**
     * @brief Callback invoked by sink plugins when space is available.
     *
     * Thread-safe. Wakes the pipeline thread via condition variable.
     * Called from sink plugin background threads.
     */
    void onSinkSpaceAvailable();

    /**
     * @brief Get event-driven wake latency statistics (microseconds)
     */
    double getAverageWakeLatencyUs() const;
    double getMaxWakeLatencyUs() const;

private:
    enum class BackpressureMode
    {
        WaitAndRetry,
        Drop,
        WriteRetry
    };

    void processingThread();
    void processAudioFrame();

    std::shared_ptr<AudioSourcePlugin> source_;
    std::shared_ptr<AudioProcessorPlugin> processor_;  // Optional (can be nullptr)
    std::shared_ptr<AudioSinkPlugin> sink_;

    std::atomic<bool> isRunning_;
    std::unique_ptr<std::thread> processingThread_;

    AudioBuffer workBuffer_;
    AudioBuffer sinkBuffer_;  // v2.2: For channel conversion (mono→stereo)
    int frameCount_;
    uint64_t processedSamples_;

    // v2.2: Channel conversion support
    int sourceChannels_ = 0;
    int sinkChannels_ = 0;

    // Sample rate adaptation (v2.0)
    int targetSampleRate_;        // Pipeline internal rate (48000 default)
    Resampler sourceResampler_;   // Source rate → 48kHz
    Resampler sinkResampler_;     // 48kHz → sink rate
    
    // v2.0: Real-time pacing and metrics
    std::chrono::steady_clock::time_point startTime_;
    uint64_t droppedSamples_;
    uint64_t driftWarnings_;
    uint64_t backpressureWaits_;
    uint64_t consecutiveFailures_;
    uint64_t processorFailures_;

    // v2.1: Drift tracking metrics
    mutable std::mutex metricsMutex_;  // Protects drift metrics
    double currentDriftMs_;
    double maxDriftMs_;
    std::atomic<uint64_t> readFailures_;
    std::atomic<uint64_t> writeFailures_;

    BackpressureMode backpressureMode_;
    int backpressureSleepMs_;
    int driftResyncMs_;
    int winTimePeriodMs_;
    bool winTimePeriodActive_;
    int longFrameWarnMs_;
    int longFrameLogIntervalMs_;
    std::chrono::steady_clock::time_point lastLongFrameLog_;

    std::atomic<float> peakLeft_;
    std::atomic<float> peakRight_;

    struct ProfilingData;
    std::unique_ptr<ProfilingData> profiling_;

    // ===== v2.2: Event-Driven Pipeline Infrastructure =====

    SchedulingMode schedulingMode_;

    // Condition variable for event-driven wake-ups
    std::condition_variable eventCV_;
    std::mutex eventMutex_;

    // Atomic flags for data/space availability (set by callbacks, cleared by pipeline)
    std::atomic<bool> sourceDataReady_;
    std::atomic<bool> sinkSpaceAvailable_;

    // Wake latency tracking
    std::chrono::steady_clock::time_point lastNotifyTime_;
    std::atomic<uint64_t> wakeLatencyTotalUs_;
    std::atomic<uint64_t> wakeLatencyMaxUs_;
    std::atomic<uint64_t> wakeCount_;

    // Event-driven timeout (fallback to prevent hangs)
    static constexpr int kEventTimeoutMs = 50;

    // Setup callbacks on async-capable plugins
    void setupEventCallbacks();
};

} // namespace nda

#endif // PROCESSINGPIPELINE_H
