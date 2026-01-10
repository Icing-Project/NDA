#include "core/ProcessingPipeline.h"
#include <thread>
#include <chrono>
#include <cstring>
#include <iostream>
#include <algorithm>
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <iomanip>
#include <sstream>
#include <ctime>
#include <string>
#include <cctype>
#include <time.h>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <mmsystem.h>
#endif

namespace nda {

namespace {

bool isTruthyEnv(const char* name)
{
    const char* value = std::getenv(name);
    if (!value) return false;

    std::string s(value);
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });

    return s == "1" || s == "true" || s == "yes" || s == "on";
}

int readEnvInt(const char* name, int defaultValue)
{
    const char* value = std::getenv(name);
    if (!value) return defaultValue;

    try {
        return std::stoi(value);
    } catch (...) {
        return defaultValue;
    }
}

uint64_t toMicros(std::chrono::steady_clock::duration d)
{
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(d).count());
}

uint64_t getCurrentThreadCpuTimeUs()
{
#ifdef _WIN32
    FILETIME creationTime{};
    FILETIME exitTime{};
    FILETIME kernelTime{};
    FILETIME userTime{};

    if (!GetThreadTimes(GetCurrentThread(), &creationTime, &exitTime, &kernelTime, &userTime)) {
        return 0;
    }

    ULARGE_INTEGER k{};
    k.LowPart = kernelTime.dwLowDateTime;
    k.HighPart = kernelTime.dwHighDateTime;

    ULARGE_INTEGER u{};
    u.LowPart = userTime.dwLowDateTime;
    u.HighPart = userTime.dwHighDateTime;

    const uint64_t total100ns = k.QuadPart + u.QuadPart;
    return total100ns / 10ULL;
#else
#ifdef CLOCK_THREAD_CPUTIME_ID
    timespec ts{};
    if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts) != 0) {
        return 0;
    }
    return static_cast<uint64_t>(ts.tv_sec) * 1000000ULL + static_cast<uint64_t>(ts.tv_nsec) / 1000ULL;
#else
    return 0;
#endif
#endif
}

std::string formatWallClockNow()
{
    using namespace std::chrono;
    const auto now = system_clock::now();
    const auto ms = duration_cast<milliseconds>(now.time_since_epoch()) % 1000;
    const std::time_t t = system_clock::to_time_t(now);

    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif

    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S") << "." << std::setw(3) << std::setfill('0') << ms.count();
    return oss.str();
}

#ifdef _WIN32
bool enableWindowsTimerPeriod(int periodMs)
{
    if (periodMs <= 0) return false;

    const MMRESULT result = timeBeginPeriod(static_cast<UINT>(periodMs));
    if (result != TIMERR_NOERROR) {
        std::cerr << "[Pipeline] timeBeginPeriod(" << periodMs
                  << ") failed (MMRESULT " << static_cast<unsigned int>(result) << ")" << std::endl;
        return false;
    }
    return true;
}

void disableWindowsTimerPeriod(int periodMs)
{
    if (periodMs <= 0) return;
    timeEndPeriod(static_cast<UINT>(periodMs));
}
#endif

} // namespace

struct ProcessingPipeline::ProfilingData
{
    struct Stat
    {
        uint64_t count = 0;
        uint64_t totalUs = 0;
        uint64_t maxUs = 0;

        void add(uint64_t us)
        {
            totalUs += us;
            count++;
            if (us > maxUs) maxUs = us;
        }

        void reset()
        {
            count = 0;
            totalUs = 0;
            maxUs = 0;
        }

        double avgUs() const
        {
            return count ? static_cast<double>(totalUs) / static_cast<double>(count) : 0.0;
        }
    };

    std::chrono::steady_clock::time_point lastLogTime{};
    std::chrono::milliseconds logInterval{1000};

    uint64_t lastProcessedSamples = 0;
    uint64_t lastDroppedSamples = 0;
    uint64_t lastDriftWarnings = 0;
    uint64_t lastBackpressureWaits = 0;

    // Stage timings (per processAudioFrame call)
    Stat frameUs{};
    Stat readUs{};
    Stat resampleInUs{};
    Stat processorUs{};
    Stat resampleOutUs{};
    Stat sinkAvailUs{};
    Stat sinkWriteUs{};
    Stat backpressureSleepUs{};

    // Thread pacing timings (per processingThread loop)
    Stat pacingSleepUs{};
    Stat pacingOvershootUs{};

    uint64_t lastThreadCpuUs = 0;

    // Counters/values (interval)
    uint64_t readFailures = 0;
    uint64_t sinkWriteFailures = 0;
    uint64_t backpressureHits = 0;
    uint64_t backpressureDrops = 0;

    uint64_t availableSamples = 0;
    uint64_t availableTotal = 0;
    int availableMin = std::numeric_limits<int>::max();
    int availableMax = 0;

    uint64_t driftSamples = 0;
    int64_t driftUsMax = 0;
    int64_t driftUsTotal = 0;

    void resetInterval(std::chrono::steady_clock::time_point now,
                       uint64_t processedSamples,
                       uint64_t droppedSamples,
                       uint64_t driftWarnings,
                       uint64_t backpressureWaits,
                       uint64_t threadCpuUs)
    {
        lastLogTime = now;
        lastProcessedSamples = processedSamples;
        lastDroppedSamples = droppedSamples;
        lastDriftWarnings = driftWarnings;
        lastBackpressureWaits = backpressureWaits;
        lastThreadCpuUs = threadCpuUs;

        frameUs.reset();
        readUs.reset();
        resampleInUs.reset();
        processorUs.reset();
        resampleOutUs.reset();
        sinkAvailUs.reset();
        sinkWriteUs.reset();
        backpressureSleepUs.reset();
        pacingSleepUs.reset();
        pacingOvershootUs.reset();

        readFailures = 0;
        sinkWriteFailures = 0;
        backpressureHits = 0;
        backpressureDrops = 0;

        availableSamples = 0;
        availableTotal = 0;
        availableMin = std::numeric_limits<int>::max();
        availableMax = 0;

        driftSamples = 0;
        driftUsMax = 0;
        driftUsTotal = 0;
    }

    void maybeLog(std::chrono::steady_clock::time_point now,
                  int frameSize,
                  int targetRate,
                  uint64_t processedSamples,
                  uint64_t droppedSamples,
                  uint64_t driftWarnings,
                  uint64_t backpressureWaits)
    {
        const uint64_t threadCpuUsNow = getCurrentThreadCpuTimeUs();

        if (lastLogTime.time_since_epoch().count() == 0) {
            resetInterval(now, processedSamples, droppedSamples, driftWarnings, backpressureWaits, threadCpuUsNow);
            return;
        }

        auto dt = now - lastLogTime;
        if (dt < logInterval) return;

        // Handle counter resets (e.g., pipeline stopped and restarted) to avoid unsigned underflow.
        if (processedSamples < lastProcessedSamples ||
            droppedSamples < lastDroppedSamples ||
            driftWarnings < lastDriftWarnings ||
            backpressureWaits < lastBackpressureWaits ||
            (threadCpuUsNow != 0 && lastThreadCpuUs != 0 && threadCpuUsNow < lastThreadCpuUs)) {
            resetInterval(now, processedSamples, droppedSamples, driftWarnings, backpressureWaits, threadCpuUsNow);
            return;
        }

        const double dtSec = std::chrono::duration_cast<std::chrono::duration<double>>(dt).count();
        const bool cpuAvailable = (threadCpuUsNow != 0) || (lastThreadCpuUs != 0);
        const uint64_t cpuDeltaUs = (cpuAvailable && threadCpuUsNow >= lastThreadCpuUs)
                                        ? (threadCpuUsNow - lastThreadCpuUs)
                                        : 0;
        const double cpuPct = (cpuAvailable && dtSec > 0.0)
                                  ? ((static_cast<double>(cpuDeltaUs) / 1000000.0) / dtSec) * 100.0
                                  : 0.0;
        const uint64_t processedDelta = processedSamples - lastProcessedSamples;
        const uint64_t droppedDelta = droppedSamples - lastDroppedSamples;
        const uint64_t attemptedDelta = processedDelta + droppedDelta;
        const uint64_t driftWarnDelta = driftWarnings - lastDriftWarnings;
        const uint64_t backpressureWaitsDelta = backpressureWaits - lastBackpressureWaits;

        const double processedSec = static_cast<double>(processedDelta) / static_cast<double>(targetRate);
        const double attemptedSec = static_cast<double>(attemptedDelta) / static_cast<double>(targetRate);
        const double droppedSec = static_cast<double>(droppedDelta) / static_cast<double>(targetRate);

        const double loopHz = dtSec > 0.0 ? static_cast<double>(frameUs.count) / dtSec : 0.0;
        const double expectedHz = frameSize > 0 ? static_cast<double>(targetRate) / static_cast<double>(frameSize) : 0.0;
        const double availAvg = availableSamples
                                    ? static_cast<double>(availableTotal) / static_cast<double>(availableSamples)
                                    : 0.0;
        const double driftAvgMs = driftSamples
                                      ? (static_cast<double>(driftUsTotal) / static_cast<double>(driftSamples)) / 1000.0
                                      : 0.0;
        const double driftMaxMs = static_cast<double>(driftUsMax) / 1000.0;

        std::cout << std::fixed << std::setprecision(2);
        std::cout << "[PipelineProfile] dt=" << (dtSec * 1000.0) << "ms"
                  << " loops=" << frameUs.count << " (" << loopHz << "Hz exp=" << expectedHz << ")"
                  << " frame=" << frameSize;
        if (cpuAvailable) {
            std::cout << " cpu=" << cpuPct << "%";
        } else {
            std::cout << " cpu=n/a";
        }
        std::cout << " processed=" << processedSec << "s"
                  << " attempted=" << attemptedSec << "s"
                  << " dropped=" << droppedSec << "s"
                  << " bpHits=" << backpressureHits
                  << " bpDrops=" << backpressureDrops
                  << " driftMax=" << driftMaxMs << "ms"
                  << " driftAvg=" << driftAvgMs << "ms";
        if (availableSamples) {
            std::cout << " avail(min/avg/max)="
                      << (availableMin == std::numeric_limits<int>::max() ? 0 : availableMin) << "/"
                      << availAvg << "/"
                      << availableMax;
        } else {
            std::cout << " avail=n/a";
        }
        std::cout << " avgUs(total/read/avail/write/bpSleep/paceSleep/paceOver)="
                  << frameUs.avgUs() << "/"
                  << readUs.avgUs() << "/"
                  << sinkAvailUs.avgUs() << "/"
                  << sinkWriteUs.avgUs() << "/"
                  << backpressureSleepUs.avgUs() << "/"
                  << pacingSleepUs.avgUs() << "/"
                  << pacingOvershootUs.avgUs()
                  << " maxUs(total/read/avail/write/paceOver)="
                  << frameUs.maxUs << "/"
                  << readUs.maxUs << "/"
                  << sinkAvailUs.maxUs << "/"
                  << sinkWriteUs.maxUs << "/"
                  << pacingOvershootUs.maxUs
                  << " failures(read/write)=" << readFailures << "/" << sinkWriteFailures
                  << " diag(driftWarn/bpWaits)="
                  << driftWarnDelta << "/"
                  << backpressureWaitsDelta
                  << std::endl;

        resetInterval(now, processedSamples, droppedSamples, driftWarnings, backpressureWaits, threadCpuUsNow);
    }
};

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
    , backpressureMode_(BackpressureMode::WaitAndRetry)
    , backpressureSleepMs_(5)
    , driftResyncMs_(250)
    , winTimePeriodMs_(0)
    , winTimePeriodActive_(false)
    , longFrameWarnMs_(0)
    , longFrameLogIntervalMs_(1000)
    , lastLongFrameLog_()
    , peakLeft_(0.0f)
    , peakRight_(0.0f)
{
    if (isTruthyEnv("NDA_PROFILE") || isTruthyEnv("NDA_PROFILE_PIPELINE")) {
        profiling_ = std::make_unique<ProfilingData>();
        const int intervalMs = std::max(100, readEnvInt("NDA_PROFILE_PIPELINE_INTERVAL_MS", 1000));
        profiling_->logInterval = std::chrono::milliseconds(intervalMs);
        std::cout << "[PipelineProfile] Enabled (interval " << intervalMs << "ms)" << std::endl;
    }

    backpressureSleepMs_ = std::max(0, readEnvInt("NDA_PIPELINE_BACKPRESSURE_SLEEP_MS", backpressureSleepMs_));
    driftResyncMs_ = std::max(0, readEnvInt("NDA_PIPELINE_DRIFT_RESYNC_MS", driftResyncMs_));
    longFrameWarnMs_ = std::max(0, readEnvInt("NDA_PIPELINE_LONG_FRAME_MS", longFrameWarnMs_));
    longFrameLogIntervalMs_ = std::max(0, readEnvInt("NDA_PIPELINE_LONG_FRAME_LOG_INTERVAL_MS", longFrameLogIntervalMs_));
    frameCount_ = std::max(64, readEnvInt("NDA_PIPELINE_FRAME_SIZE", frameCount_));

#ifdef _WIN32
    winTimePeriodMs_ = std::max(0, readEnvInt("NDA_PIPELINE_WIN_TIME_PERIOD_MS", 1));
#else
    winTimePeriodMs_ = 0;
#endif

    if (const char* mode = std::getenv("NDA_PIPELINE_BACKPRESSURE_MODE")) {
        std::string s(mode);
        s.erase(std::remove_if(s.begin(), s.end(), [](unsigned char c) { return std::isspace(c) != 0; }), s.end());
        std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });

        if (s == "wait" || s == "wait_retry" || s == "waitandretry" || s == "default") {
            backpressureMode_ = BackpressureMode::WaitAndRetry;
        } else if (s == "drop") {
            backpressureMode_ = BackpressureMode::Drop;
        } else if (s == "write" || s == "write_retry" || s == "writeretry" || s == "retry") {
            backpressureMode_ = BackpressureMode::WriteRetry;
        } else {
            std::cerr << "[Pipeline] Unknown NDA_PIPELINE_BACKPRESSURE_MODE='" << mode
                      << "' (expected wait|drop|write_retry); using default wait" << std::endl;
        }
    }

    const char* modeLabel = "wait";
    if (backpressureMode_ == BackpressureMode::Drop) modeLabel = "drop";
    if (backpressureMode_ == BackpressureMode::WriteRetry) modeLabel = "write_retry";
    std::cout << "[Pipeline] Backpressure mode: " << modeLabel
              << " (sleep " << backpressureSleepMs_ << "ms)"
              << ", driftResync=" << driftResyncMs_ << "ms"
#ifdef _WIN32
              << ", winTimePeriod=" << winTimePeriodMs_ << "ms"
#endif
              << ", longFrameWarn=" << longFrameWarnMs_ << "ms"
              << std::endl;
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
    const int sourceRate = source_->getSampleRate();
    const int sinkRate = sink_->getSampleRate();
    const int channels = source_->getChannels();

    const int requestedFrames = std::max(64, frameCount_);
    source_->setBufferSize(requestedFrames);
    sink_->setBufferSize(requestedFrames);

    int sourceFrames = source_->getBufferSize();
    if (sourceFrames <= 0) sourceFrames = requestedFrames;
    int sinkFrames = sink_->getBufferSize();
    if (sinkFrames <= 0) sinkFrames = requestedFrames;
    int negotiatedFrames = std::min(requestedFrames, std::min(sourceFrames, sinkFrames));
    if (negotiatedFrames <= 0) {
        negotiatedFrames = requestedFrames;
    }

    if (negotiatedFrames != requestedFrames || sourceFrames != requestedFrames || sinkFrames != requestedFrames) {
        std::cout << "[Pipeline] Frame size negotiated: requested=" << requestedFrames
                  << ", source=" << sourceFrames
                  << ", sink=" << sinkFrames
                  << ", using=" << negotiatedFrames
                  << std::endl;
    }

    if (negotiatedFrames != requestedFrames) {
        source_->setBufferSize(negotiatedFrames);
        sink_->setBufferSize(negotiatedFrames);
    }

    frameCount_ = negotiatedFrames;
    
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

#ifdef _WIN32
    bool enabledWinTimerThisStart = false;
    if (!winTimePeriodActive_ && winTimePeriodMs_ > 0) {
        if (enableWindowsTimerPeriod(winTimePeriodMs_)) {
            winTimePeriodActive_ = true;
            enabledWinTimerThisStart = true;
            std::cout << "[Pipeline] Windows timer period active (" << winTimePeriodMs_ << "ms)" << std::endl;
        }
    }

    auto rollbackWinTimer = [&]() {
        if (enabledWinTimerThisStart) {
            disableWindowsTimerPeriod(winTimePeriodMs_);
            winTimePeriodActive_ = false;
        }
    };
#endif

    // Start all plugins (3-slot model)
    if (!source_->start()) {
        std::cerr << "[Pipeline] Source start failed" << std::endl;
#ifdef _WIN32
        rollbackWinTimer();
#endif
        return false;
    }
    
    if (processor_ && !processor_->start()) {
        std::cerr << "[Pipeline] Processor start failed" << std::endl;
        source_->stop();  // Rollback
#ifdef _WIN32
        rollbackWinTimer();
#endif
        return false;
    }
    
    if (!sink_->start()) {
        std::cerr << "[Pipeline] Sink start failed" << std::endl;
        if (processor_) processor_->stop();
        source_->stop();  // Rollback
#ifdef _WIN32
        rollbackWinTimer();
#endif
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
    lastLongFrameLog_ = std::chrono::steady_clock::time_point{};
    if (profiling_) {
        profiling_->resetInterval(startTime_, processedSamples_, droppedSamples_, driftWarnings_, backpressureWaits_, 0);
    }

    std::cout << "[Pipeline] Start @ " << formatWallClockNow() << std::endl;
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

    std::cout << "[Pipeline] Stopping @ " << formatWallClockNow()
              << "... (processed " << processedSamples_ << " samples so far)" << std::endl;

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

    const auto wallMs = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - startTime_
    ).count();
    const double processedSec = static_cast<double>(processedSamples_) / static_cast<double>(targetSampleRate_);
    const double attemptedSec = static_cast<double>(processedSamples_ + droppedSamples_) / static_cast<double>(targetSampleRate_);

    std::cout << "[Pipeline] Stop @ " << formatWallClockNow()
              << " wall=" << (static_cast<double>(wallMs) / 1000.0)
              << "s processed=" << processedSec << "s attempted=" << attemptedSec << "s"
              << std::endl;

#ifdef _WIN32
    if (winTimePeriodActive_) {
        disableWindowsTimerPeriod(winTimePeriodMs_);
        winTimePeriodActive_ = false;
        std::cout << "[Pipeline] Windows timer period released" << std::endl;
    }
#endif
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
    uint64_t driftResyncEvents = 0;

    while (isRunning_) {
        processAudioFrame();
        frameCount++;

        // v2.0: Real-time pacing - sleep to maintain exact 1.0x real-time cadence
        uint64_t attemptedSamples = processedSamples_ + droppedSamples_;
        auto targetTime = startTime_ + std::chrono::microseconds(
            (attemptedSamples * 1000000) / targetSampleRate_
        );

        auto now = std::chrono::steady_clock::now();

        if (now < targetTime) {
            // We're ahead of schedule - sleep until target time
            auto sleepStart = now;
            std::this_thread::sleep_until(targetTime);
            const auto afterSleep = std::chrono::steady_clock::now();
            if (profiling_) {
                profiling_->pacingSleepUs.add(toMicros(afterSleep - sleepStart));
                if (afterSleep > targetTime) {
                    profiling_->pacingOvershootUs.add(toMicros(afterSleep - targetTime));
                }
            }
            now = afterSleep;
        } else {
            // We're behind schedule - track drift
            auto drift = std::chrono::duration_cast<std::chrono::milliseconds>(now - targetTime);
            if (profiling_) {
                const int64_t driftUs = static_cast<int64_t>(toMicros(now - targetTime));
                profiling_->driftSamples++;
                profiling_->driftUsTotal += driftUs;
                if (driftUs > profiling_->driftUsMax) profiling_->driftUsMax = driftUs;
            }
            if (drift.count() > 50) {
                // More than 50ms behind - log warning
                driftWarnings_++;
                if (driftWarnings_ % 100 == 0) {
                    std::cerr << "[Pipeline] WARNING: " << drift.count() 
                              << "ms behind schedule (drift #" << driftWarnings_ << ")" << std::endl;
                }
            }

            // v2.1: DISABLED drift resync - it drops audio which violates zero-loss requirement
            // Previously this would artificially increment droppedSamples_ to "catch up" when behind schedule.
            // This caused audible artifacts (clicks, pops, crackling).
            // New strategy: Let pipeline run as fast as it can. If it falls behind, that's a plugin performance
            // issue that should be fixed, not hidden by dropping audio.

            // Original code (DISABLED):
            // if (driftResyncMs_ > 0 && drift.count() > driftResyncMs_) {
            //     droppedSamples_ += catchUpSamples;  // This was dropping audio!
            // }
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

        if (profiling_) {
            profiling_->maybeLog(
                std::chrono::steady_clock::now(),
                frameCount_,
                targetSampleRate_,
                processedSamples_,
                droppedSamples_,
                driftWarnings_,
                backpressureWaits_
            );
        }
    }

    std::cout << "[Pipeline] Processing thread exiting" << std::endl;
}

// v2.1: Safe plugin call wrapper with exception handling
// Prevents plugin crashes from taking down the entire pipeline
template<typename Func>
bool safePluginCall(const char* pluginName, const char* operation, Func&& func, std::atomic<bool>& isRunning)
{
    try {
        return func();
    } catch (const std::exception& e) {
        std::cerr << "[Pipeline] CRITICAL: Plugin '" << pluginName << "' threw exception during "
                  << operation << ": " << e.what() << std::endl;
        std::cerr << "[Pipeline] Failing pipeline to prevent corruption." << std::endl;
        isRunning.store(false);
        return false;
    } catch (...) {
        std::cerr << "[Pipeline] CRITICAL: Plugin '" << pluginName << "' crashed during "
                  << operation << " (unknown exception)" << std::endl;
        std::cerr << "[Pipeline] Failing pipeline to prevent corruption." << std::endl;
        isRunning.store(false);
        return false;
    }
}

void ProcessingPipeline::processAudioFrame()
{
    static int frameDebugCount = 0;
    if (++frameDebugCount <= 5) {
        std::cout << "[Pipeline] processAudioFrame #" << frameDebugCount
                  << " isRunning=" << isRunning_.load() << std::endl;
    }

    if (!source_ || !sink_) return;

    auto frameStart = std::chrono::steady_clock::now();
    auto readStart = frameStart;
    uint64_t readUs = 0;
    uint64_t resampleInUs = 0;
    uint64_t processorUs = 0;
    uint64_t resampleOutUs = 0;
    uint64_t sinkAvailUs = 0;
    uint64_t sinkWriteUs = 0;
    uint64_t backpressureSleepUs = 0;
    int lastAvailableFrames = -1;

    auto maybeLogLongFrame = [&](const char* reason) {
        if (longFrameWarnMs_ <= 0) return;

        const auto now = std::chrono::steady_clock::now();
        if (lastLongFrameLog_.time_since_epoch().count() != 0) {
            const auto minDt = std::chrono::milliseconds(longFrameLogIntervalMs_);
            if (now - lastLongFrameLog_ < minDt) return;
        }

        const uint64_t totalUs = toMicros(now - frameStart);
        const uint64_t warnUs = static_cast<uint64_t>(longFrameWarnMs_) * 1000ULL;
        if (totalUs < warnUs) return;

        const uint64_t expectedUs = (static_cast<uint64_t>(frameCount_) * 1000000ULL)
                                        / static_cast<uint64_t>(std::max(1, targetSampleRate_));

        lastLongFrameLog_ = now;
        std::cerr << "[Pipeline] Long frame (" << reason << "): total=" << totalUs
                  << "us expected=" << expectedUs
                  << "us read=" << readUs
                  << " resIn=" << resampleInUs
                  << " proc=" << processorUs
                  << " resOut=" << resampleOutUs
                  << " avail=" << sinkAvailUs
                  << " write=" << sinkWriteUs
                  << " bpSleep=" << backpressureSleepUs;
        if (lastAvailableFrames >= 0) {
            std::cerr << " availFrames=" << lastAvailableFrames;
        } else {
            std::cerr << " availFrames=n/a";
        }
        std::cerr << " frame=" << frameCount_
                  << " dropped=" << droppedSamples_
                  << std::endl;
    };

    // 1. Read audio from source (with exception safety)
    bool readOk = safePluginCall(source_->getInfo().name.c_str(), "readAudio",
                                   [&]() { return source_->readAudio(workBuffer_); }, isRunning_);

    static int debugReadCount = 0;
    if (++debugReadCount <= 5) {
        std::cout << "[Pipeline] Read #" << debugReadCount << " readOk=" << readOk
                  << " isRunning=" << isRunning_.load() << std::endl;
    }

    if (!readOk) {
        consecutiveFailures_++;
        readUs = toMicros(std::chrono::steady_clock::now() - readStart);
        if (profiling_) {
            profiling_->readUs.add(readUs);
            profiling_->readFailures++;
            profiling_->frameUs.add(toMicros(std::chrono::steady_clock::now() - frameStart));
        }

        if (consecutiveFailures_ == 1) {
            std::cerr << "[Pipeline] Audio read started failing" << std::endl;
        }

        if (consecutiveFailures_ > 10 && consecutiveFailures_ % 100 == 0) {
            std::cerr << "[Pipeline] " << consecutiveFailures_ << " consecutive read failures" << std::endl;
        }

        // v2.2: Allow current frame to complete even if stop() was called mid-frame
        // Removed early exit here - let the frame finish writing to avoid buffer underruns

        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        maybeLogLongFrame("read_fail");
        return;
    }

    readUs = toMicros(std::chrono::steady_clock::now() - readStart);
    if (profiling_) profiling_->readUs.add(readUs);

    // Reset failure counter on success
    if (consecutiveFailures_ > 0) {
        std::cerr << "[Pipeline] Audio read recovered after " << consecutiveFailures_ << " failures" << std::endl;
        consecutiveFailures_ = 0;
    }

    // 2. Resample source → 48kHz (auto-applied if source rate differs)
    if (sourceResampler_.isActive()) {
        auto t0 = std::chrono::steady_clock::now();
        sourceResampler_.process(workBuffer_);
        resampleInUs = toMicros(std::chrono::steady_clock::now() - t0);
        if (profiling_) {
            profiling_->resampleInUs.add(resampleInUs);
        }
    }

    // 3. Apply processor at 48kHz (optional - encryptor, decryptor, effects, etc.)
    if (processor_ && processor_->getState() == PluginState::Running) {
        auto t0 = std::chrono::steady_clock::now();

        // v2.1: Safe plugin call with fail-fast on critical failure
        bool processOk = safePluginCall(processor_->getInfo().name.c_str(), "processAudio",
                                         [&]() { return processor_->processAudio(workBuffer_); }, isRunning_);

        processorUs = toMicros(std::chrono::steady_clock::now() - t0);
        if (profiling_) {
            profiling_->processorUs.add(processorUs);
        }

        if (!processOk) {
            // v2.1: FAIL-FAST on processor failure
            // Reason: Passthrough could send unencrypted/unprocessed audio, violating security
            processorFailures_++;
            std::cerr << "[Pipeline] CRITICAL: Processor failed after " << processorFailures_
                      << " failures. Stopping pipeline to prevent unprocessed audio transmission." << std::endl;
            isRunning_.store(false);
            return;
        }
    }

    // 4. Resample 48kHz → sink rate (auto-applied if sink rate differs)
    if (sinkResampler_.isActive()) {
        auto t0 = std::chrono::steady_clock::now();
        sinkResampler_.process(workBuffer_);
        resampleOutUs = toMicros(std::chrono::steady_clock::now() - t0);
        if (profiling_) {
            profiling_->resampleOutUs.add(resampleOutUs);
        }
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

    const int frameCount = workBuffer_.getFrameCount();

    // 5/6. Backpressure + write (ZERO-LOSS MODE)
    // v2.1: Never drop audio - wait for sink with latency budget enforcement
    bool writeOk = false;
    const int maxLatencyMs = 50;  // Fail-fast if we exceed latency budget
    auto backpressureStart = std::chrono::steady_clock::now();
    int retryAttempts = 0;
    const int maxRetries = 20;  // At 5ms sleep, this is ~100ms total

    while (!writeOk && retryAttempts < maxRetries) {
        // Check available space (with exception safety)
        auto availStart = std::chrono::steady_clock::now();
        int available = 0;
        bool availOk = safePluginCall(sink_->getInfo().name.c_str(), "getAvailableSpace",
                                       [&]() { available = sink_->getAvailableSpace(); return true; }, isRunning_);
        if (!availOk) {
            // Critical failure in getAvailableSpace
            return;
        }

        lastAvailableFrames = available;
        const uint64_t availUs1 = toMicros(std::chrono::steady_clock::now() - availStart);
        sinkAvailUs += availUs1;
        if (profiling_) {
            profiling_->sinkAvailUs.add(availUs1);
            profiling_->availableSamples++;
            profiling_->availableTotal += static_cast<uint64_t>(std::max(0, available));
            profiling_->availableMin = std::min(profiling_->availableMin, available);
            profiling_->availableMax = std::max(profiling_->availableMax, available);
        }

        // Try to write (with exception safety)
        auto writeStart = std::chrono::steady_clock::now();
        writeOk = safePluginCall(sink_->getInfo().name.c_str(), "writeAudio",
                                  [&]() { return sink_->writeAudio(workBuffer_); }, isRunning_);
        sinkWriteUs += toMicros(std::chrono::steady_clock::now() - writeStart);
        if (profiling_) {
            profiling_->sinkWriteUs.add(toMicros(std::chrono::steady_clock::now() - writeStart));
        }

        // v2.2: Allow current frame to complete even if stop() was called mid-frame
        // Removed early exit here - let the frame finish writing to avoid audio artifacts

        if (writeOk) {
            break;  // Success!
        }

        // Write failed - check if we've exceeded latency budget
        auto cumulativeWaitMs = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - backpressureStart
        ).count();

        if (cumulativeWaitMs >= maxLatencyMs) {
            // FAIL-FAST: We've exceeded our latency budget
            std::cerr << "[Pipeline] CRITICAL: Sink cannot keep up - exceeded " << maxLatencyMs
                      << "ms latency budget after " << retryAttempts << " retries. "
                      << "Failing pipeline to prevent audio corruption." << std::endl;
            isRunning_.store(false);  // Stop pipeline
            if (profiling_) {
                profiling_->frameUs.add(toMicros(std::chrono::steady_clock::now() - frameStart));
            }
            maybeLogLongFrame("latency_budget_exceeded");
            return;
        }

        // Backpressure detected - wait and retry
        backpressureWaits_++;
        if (profiling_) profiling_->backpressureHits++;

        if (retryAttempts == 0) {
            std::cerr << "[Pipeline] Sink backpressure detected, waiting (available: "
                      << available << ", needed: " << frameCount << ")" << std::endl;
        }

        // Sleep before retry
        if (backpressureSleepMs_ > 0) {
            auto sleepStart = std::chrono::steady_clock::now();
            std::this_thread::sleep_for(std::chrono::milliseconds(backpressureSleepMs_));
            const uint64_t sleepUs = toMicros(std::chrono::steady_clock::now() - sleepStart);
            backpressureSleepUs += sleepUs;
            if (profiling_) {
                profiling_->backpressureSleepUs.add(sleepUs);
            }
        }

        retryAttempts++;
    }

    // If we exhausted retries without success, fail pipeline
    if (!writeOk) {
        std::cerr << "[Pipeline] CRITICAL: Sink write failed after " << maxRetries
                  << " retries. Failing pipeline." << std::endl;
        isRunning_.store(false);
        if (profiling_) {
            profiling_->sinkWriteFailures++;
            profiling_->frameUs.add(toMicros(std::chrono::steady_clock::now() - frameStart));
        }
        maybeLogLongFrame("write_exhausted");
        return;
    }

    // Success - update metrics
    processedSamples_ += frameCount;

    if (profiling_) {
        profiling_->frameUs.add(toMicros(std::chrono::steady_clock::now() - frameStart));
    }

    maybeLogLongFrame("ok");
}

} // namespace nda
