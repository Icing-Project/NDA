#ifndef PLUGINTYPES_H
#define PLUGINTYPES_H

namespace nda {

enum class PluginType {
    AudioSource,    // Audio input (microphone, file, streaming service)
    AudioSink,      // Audio output (speakers, file, network)
    Processor       // Audio transformation (encryption, effects, resampling, etc.)
};

enum class PluginState {
    Unloaded,
    Loaded,
    Initialized,
    Running,
    Error
};

} // namespace nda

#endif // PLUGINTYPES_H
