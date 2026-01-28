#include "audio/PulseDeviceEnum.h"

#ifdef __linux__
#include <pulse/pulseaudio.h>
#include <pulse/error.h>
#include <iostream>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <algorithm>

namespace nda {

namespace {

// Helper class for synchronous PulseAudio operations
class PulseEnumerator {
public:
    PulseEnumerator() : mainloop_(nullptr), context_(nullptr), ready_(false), done_(false) {}

    ~PulseEnumerator() {
        cleanup();
    }

    bool connect() {
        mainloop_ = pa_mainloop_new();
        if (!mainloop_) {
            std::cerr << "[PulseDeviceEnum] Failed to create mainloop\n";
            return false;
        }

        pa_mainloop_api* api = pa_mainloop_get_api(mainloop_);
        context_ = pa_context_new(api, "NDA Device Enumerator");
        if (!context_) {
            std::cerr << "[PulseDeviceEnum] Failed to create context\n";
            return false;
        }

        pa_context_set_state_callback(context_, contextStateCallback, this);

        if (pa_context_connect(context_, nullptr, PA_CONTEXT_NOFLAGS, nullptr) < 0) {
            std::cerr << "[PulseDeviceEnum] Failed to connect: "
                      << pa_strerror(pa_context_errno(context_)) << "\n";
            return false;
        }

        // Run mainloop until connected or failed
        while (!ready_ && !done_) {
            if (pa_mainloop_iterate(mainloop_, 1, nullptr) < 0) {
                break;
            }
        }

        return ready_;
    }

    std::vector<PulseDeviceInfo> getSources() {
        sources_.clear();
        done_ = false;

        pa_operation* op = pa_context_get_source_info_list(
            context_, sourceInfoCallback, this);

        if (op) {
            waitForOperation(op);
            pa_operation_unref(op);
        }

        return sources_;
    }

    std::vector<PulseDeviceInfo> getSinks() {
        sinks_.clear();
        done_ = false;

        pa_operation* op = pa_context_get_sink_info_list(
            context_, sinkInfoCallback, this);

        if (op) {
            waitForOperation(op);
            pa_operation_unref(op);
        }

        return sinks_;
    }

    std::string getDefaultSource() {
        defaultSource_.clear();
        done_ = false;

        pa_operation* op = pa_context_get_server_info(
            context_, serverInfoCallback, this);

        if (op) {
            waitForOperation(op);
            pa_operation_unref(op);
        }

        return defaultSource_;
    }

    std::string getDefaultSink() {
        defaultSink_.clear();
        done_ = false;

        pa_operation* op = pa_context_get_server_info(
            context_, serverInfoCallback, this);

        if (op) {
            waitForOperation(op);
            pa_operation_unref(op);
        }

        return defaultSink_;
    }

private:
    pa_mainloop* mainloop_;
    pa_context* context_;
    bool ready_;
    bool done_;

    std::vector<PulseDeviceInfo> sources_;
    std::vector<PulseDeviceInfo> sinks_;
    std::string defaultSource_;
    std::string defaultSink_;

    void cleanup() {
        if (context_) {
            pa_context_disconnect(context_);
            pa_context_unref(context_);
            context_ = nullptr;
        }
        if (mainloop_) {
            pa_mainloop_free(mainloop_);
            mainloop_ = nullptr;
        }
    }

    void waitForOperation(pa_operation* op) {
        while (pa_operation_get_state(op) == PA_OPERATION_RUNNING) {
            if (pa_mainloop_iterate(mainloop_, 1, nullptr) < 0) {
                break;
            }
        }
    }

    static void contextStateCallback(pa_context* c, void* userdata) {
        auto* self = static_cast<PulseEnumerator*>(userdata);
        pa_context_state_t state = pa_context_get_state(c);

        switch (state) {
            case PA_CONTEXT_READY:
                self->ready_ = true;
                break;
            case PA_CONTEXT_FAILED:
            case PA_CONTEXT_TERMINATED:
                self->done_ = true;
                break;
            default:
                break;
        }
    }

    static void sourceInfoCallback(pa_context* /*c*/, const pa_source_info* info,
                                   int eol, void* userdata) {
        auto* self = static_cast<PulseEnumerator*>(userdata);

        if (eol > 0) {
            self->done_ = true;
            return;
        }

        if (!info) return;

        // Skip monitor sources (they capture output audio, not input)
        if (info->monitor_of_sink != PA_INVALID_INDEX) {
            return;
        }

        PulseDeviceInfo device;
        device.name = info->name ? info->name : "";
        device.description = info->description ? info->description : device.name;
        device.index = info->index;
        device.isDefault = false;  // Will be set later
        device.sampleRate = info->sample_spec.rate;
        device.channels = info->sample_spec.channels;

        self->sources_.push_back(device);
    }

    static void sinkInfoCallback(pa_context* /*c*/, const pa_sink_info* info,
                                 int eol, void* userdata) {
        auto* self = static_cast<PulseEnumerator*>(userdata);

        if (eol > 0) {
            self->done_ = true;
            return;
        }

        if (!info) return;

        PulseDeviceInfo device;
        device.name = info->name ? info->name : "";
        device.description = info->description ? info->description : device.name;
        device.index = info->index;
        device.isDefault = false;  // Will be set later
        device.sampleRate = info->sample_spec.rate;
        device.channels = info->sample_spec.channels;

        self->sinks_.push_back(device);
    }

    static void serverInfoCallback(pa_context* /*c*/, const pa_server_info* info,
                                   void* userdata) {
        auto* self = static_cast<PulseEnumerator*>(userdata);

        if (info) {
            self->defaultSource_ = info->default_source_name ? info->default_source_name : "";
            self->defaultSink_ = info->default_sink_name ? info->default_sink_name : "";
        }

        self->done_ = true;
    }
};

} // anonymous namespace

std::vector<PulseDeviceInfo> enumeratePulseSources() {
    PulseEnumerator enumerator;

    if (!enumerator.connect()) {
        return {};
    }

    std::string defaultSource = enumerator.getDefaultSource();
    std::vector<PulseDeviceInfo> sources = enumerator.getSources();

    // Mark default device
    for (auto& source : sources) {
        if (source.name == defaultSource) {
            source.isDefault = true;
        }
    }

    // Sort: default first, then alphabetically by description
    std::sort(sources.begin(), sources.end(), [](const PulseDeviceInfo& a, const PulseDeviceInfo& b) {
        if (a.isDefault != b.isDefault) return a.isDefault > b.isDefault;
        return a.description < b.description;
    });

    return sources;
}

std::vector<PulseDeviceInfo> enumeratePulseSinks() {
    PulseEnumerator enumerator;

    if (!enumerator.connect()) {
        return {};
    }

    std::string defaultSink = enumerator.getDefaultSink();
    std::vector<PulseDeviceInfo> sinks = enumerator.getSinks();

    // Mark default device
    for (auto& sink : sinks) {
        if (sink.name == defaultSink) {
            sink.isDefault = true;
        }
    }

    // Sort: default first, then alphabetically by description
    std::sort(sinks.begin(), sinks.end(), [](const PulseDeviceInfo& a, const PulseDeviceInfo& b) {
        if (a.isDefault != b.isDefault) return a.isDefault > b.isDefault;
        return a.description < b.description;
    });

    return sinks;
}

std::string getDefaultPulseSource() {
    PulseEnumerator enumerator;

    if (!enumerator.connect()) {
        return "";
    }

    return enumerator.getDefaultSource();
}

std::string getDefaultPulseSink() {
    PulseEnumerator enumerator;

    if (!enumerator.connect()) {
        return "";
    }

    return enumerator.getDefaultSink();
}

bool isPulseAudioAvailable() {
    PulseEnumerator enumerator;
    return enumerator.connect();
}

// ==================== AIOC-Specific Functions ====================

namespace {

/// Helper to convert string to lowercase
std::string toLowerAIOC(const std::string& str) {
    std::string result = str;
    std::transform(result.begin(), result.end(), result.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return result;
}

/// Check if string contains "aioc" (case-insensitive)
bool containsAIOC(const std::string& str) {
    return toLowerAIOC(str).find("aioc") != std::string::npos;
}

} // anonymous namespace

std::vector<PulseDeviceInfo> findAIOCPulseSources() {
    auto allSources = enumeratePulseSources();
    std::vector<PulseDeviceInfo> aiocSources;

    for (const auto& source : allSources) {
        if (containsAIOC(source.name) || containsAIOC(source.description)) {
            aiocSources.push_back(source);
        }
    }

    return aiocSources;
}

std::vector<PulseDeviceInfo> findAIOCPulseSinks() {
    auto allSinks = enumeratePulseSinks();
    std::vector<PulseDeviceInfo> aiocSinks;

    for (const auto& sink : allSinks) {
        if (containsAIOC(sink.name) || containsAIOC(sink.description)) {
            aiocSinks.push_back(sink);
        }
    }

    return aiocSinks;
}

std::string getFirstAIOCSource() {
    auto aiocSources = findAIOCPulseSources();
    if (!aiocSources.empty()) {
        return aiocSources.front().name;
    }
    return "";
}

std::string getFirstAIOCSink() {
    auto aiocSinks = findAIOCPulseSinks();
    if (!aiocSinks.empty()) {
        return aiocSinks.front().name;
    }
    return "";
}

bool isAIOCAudioPresent() {
    return !findAIOCPulseSources().empty() || !findAIOCPulseSinks().empty();
}

} // namespace nda

#else // !__linux__

// Stub implementations for non-Linux platforms
namespace nda {

std::vector<PulseDeviceInfo> enumeratePulseSources() {
    return {};
}

std::vector<PulseDeviceInfo> enumeratePulseSinks() {
    return {};
}

std::string getDefaultPulseSource() {
    return "";
}

std::string getDefaultPulseSink() {
    return "";
}

bool isPulseAudioAvailable() {
    return false;
}

std::vector<PulseDeviceInfo> findAIOCPulseSources() {
    return {};
}

std::vector<PulseDeviceInfo> findAIOCPulseSinks() {
    return {};
}

std::string getFirstAIOCSource() {
    return "";
}

std::string getFirstAIOCSink() {
    return "";
}

bool isAIOCAudioPresent() {
    return false;
}

} // namespace nda

#endif // __linux__
