#ifndef AES256ENCRYPTORPLUGIN_H
#define AES256ENCRYPTORPLUGIN_H

#include "plugins/EncryptorPlugin.h"
#include <vector>

namespace NADE {

class AES256EncryptorPlugin : public EncryptorPlugin {
public:
    AES256EncryptorPlugin();
    ~AES256EncryptorPlugin() override;

    // BasePlugin interface
    bool initialize() override;
    void shutdown() override;
    bool start() override;
    void stop() override;
    PluginInfo getInfo() const override;
    PluginState getState() const override;
    void setParameter(const std::string& key, const std::string& value) override;
    std::string getParameter(const std::string& key) const override;

    // EncryptorPlugin interface
    bool encrypt(const uint8_t* input, size_t inputSize,
                uint8_t* output, size_t& outputSize,
                const uint8_t* nonce, size_t nonceSize) override;

    bool decrypt(const uint8_t* input, size_t inputSize,
                uint8_t* output, size_t& outputSize,
                const uint8_t* nonce, size_t nonceSize) override;

    bool setKey(const std::vector<uint8_t>& key) override;
    bool generateKey() override;
    std::vector<uint8_t> getKey() const override { return key_; }

    EncryptionAlgorithm getAlgorithm() const override { return EncryptionAlgorithm::AES_256_GCM; }
    int getKeySize() const override { return 32; } // 256 bits
    int getBlockSize() const override { return 16; }
    int getTagSize() const override { return 16; }

    bool isHardwareAccelerated() const override { return hardwareAccelerated_; }

private:
    PluginState state_;
    std::vector<uint8_t> key_;
    bool hardwareAccelerated_;
    void* cryptoContext_; // OpenSSL context
};

} // namespace NADE

NADE_DECLARE_PLUGIN(NADE::AES256EncryptorPlugin)

#endif // AES256ENCRYPTORPLUGIN_H
