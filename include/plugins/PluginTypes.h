#ifndef PLUGINTYPES_H
#define PLUGINTYPES_H

namespace NADE {

enum class PluginType {
    AudioSource,    // Audio input (microphone, file, streaming service)
    AudioSink,      // Audio output (speakers, file, network)
    Bearer,         // Network/transport simulation (TCP, UDP, custom protocols)
    Encryptor,      // Encryption/decryption module
    Processor       // Generic audio processor (effects, filters)
};

enum class PluginState {
    Unloaded,
    Loaded,
    Initialized,
    Running,
    Error
};

} // namespace NADE

#endif // PLUGINTYPES_H
