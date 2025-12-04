#ifndef ENCRYPTORPLUGIN_H
#define ENCRYPTORPLUGIN_H

#include "BasePlugin.h"
#include <vector>
#include <cstdint>

namespace nda {

enum class EncryptionAlgorithm {
    AES_128_GCM,
    AES_192_GCM,
    AES_256_GCM,
    CHACHA20_POLY1305,
    CUSTOM
};

class EncryptorPlugin : public BasePlugin {
public:
    virtual ~EncryptorPlugin() = default;

    PluginType getType() const override { return PluginType::Encryptor; }

    // Encryption methods
    virtual bool encrypt(const uint8_t* input, size_t inputSize,
                        uint8_t* output, size_t& outputSize,
                        const uint8_t* nonce, size_t nonceSize) = 0;

    virtual bool decrypt(const uint8_t* input, size_t inputSize,
                        uint8_t* output, size_t& outputSize,
                        const uint8_t* nonce, size_t nonceSize) = 0;

    // Key management
    virtual bool setKey(const std::vector<uint8_t>& key) = 0;
    virtual bool generateKey() = 0;
    virtual std::vector<uint8_t> getKey() const = 0;

    // Algorithm info
    virtual EncryptionAlgorithm getAlgorithm() const = 0;
    virtual int getKeySize() const = 0;
    virtual int getBlockSize() const = 0;
    virtual int getTagSize() const = 0;

    // Performance
    virtual bool isHardwareAccelerated() const = 0;
};

} // namespace nda

#endif // ENCRYPTORPLUGIN_H
