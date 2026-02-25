/**
 * AES-256-CTR Encryptor Processor Plugin
 *
 * Symmetric encryption using OpenSSL EVP API with AES-256 in CTR mode.
 * The cipher context is initialized once in start() with an all-zero IV and
 * then used as a streaming cipher across processAudio() calls. This makes the
 * encryption byte-position-based rather than per-buffer, so it works correctly
 * regardless of the buffer size the pipeline negotiates.
 *
 * The matching decryptor must use the same key and also start its keystream at
 * position 0 (i.e. be started at the same time with no leading data consumed).
 *
 * WARNING: Key exchange is out of scope. Keys must be shared out-of-band
 * or via the NDA Crypto menu (Generate AES Key -> Apply to Pipeline).
 */

#include "plugins/AudioProcessorPlugin.h"
#include "plugins/PluginTypes.h"
#include <openssl/evp.h>
#include <openssl/rand.h>
#include <vector>
#include <cstring>
#include <iostream>

namespace nda {

class AES256EncryptorPlugin : public AudioProcessorPlugin {
private:
    EVP_CIPHER_CTX* ctx_;
    std::vector<uint8_t> key_;     // 256-bit encryption key
    int sampleRate_;
    int channels_;
    PluginState state_;

public:
    AES256EncryptorPlugin()
        : ctx_(nullptr)
        , key_(32, 0)
        , sampleRate_(48000)
        , channels_(2)
        , state_(PluginState::Unloaded)
    {
        ctx_ = EVP_CIPHER_CTX_new();
    }

    ~AES256EncryptorPlugin() {
        if (ctx_) {
            EVP_CIPHER_CTX_free(ctx_);
            ctx_ = nullptr;
        }
    }

    bool initialize() override {
        if (!ctx_) {
            std::cerr << "[AES256Encryptor] Failed to create cipher context" << std::endl;
            state_ = PluginState::Error;
            return false;
        }

        // Generate random key (can be overridden via setParameter or Crypto menu)
        if (RAND_bytes(key_.data(), static_cast<int>(key_.size())) != 1) {
            std::cerr << "[AES256Encryptor] Failed to generate random key" << std::endl;
            state_ = PluginState::Error;
            return false;
        }

        state_ = PluginState::Initialized;
        std::cout << "[AES256Encryptor] Initialized with " << (key_.size() * 8)
                  << "-bit key" << std::endl;
        return true;
    }

    bool start() override {
        if (state_ != PluginState::Initialized) {
            return false;
        }

        // Initialize streaming CTR context once with all-zero IV.
        // All subsequent processAudio() calls use EVP_EncryptUpdate only,
        // maintaining continuous keystream position regardless of buffer size.
        uint8_t iv[16] = {0};
        if (EVP_EncryptInit_ex(ctx_, EVP_aes_256_ctr(), nullptr,
                               key_.data(), iv) != 1) {
            std::cerr << "[AES256Encryptor] Failed to initialize CTR context" << std::endl;
            state_ = PluginState::Error;
            return false;
        }

        state_ = PluginState::Running;
        std::cout << "[AES256Encryptor] Started (streaming CTR, IV=0)" << std::endl;
        return true;
    }

    void stop() override {
        if (state_ == PluginState::Running) {
            state_ = PluginState::Initialized;
            std::cout << "[AES256Encryptor] Stopped" << std::endl;
        }
    }

    void shutdown() override {
        state_ = PluginState::Unloaded;
        key_.assign(32, 0);  // Clear key from memory
        std::cout << "[AES256Encryptor] Shutdown" << std::endl;
    }

    bool processAudio(AudioBuffer& buffer) override {
        if (state_ != PluginState::Running || !ctx_) {
            return false;
        }

        // Convert float audio to bytes (interleaved layout)
        size_t totalSamples = static_cast<size_t>(buffer.getFrameCount()) * buffer.getChannelCount();
        size_t dataSize = totalSamples * sizeof(float);
        std::vector<uint8_t> plaintext(dataSize);

        float* floatPtr = reinterpret_cast<float*>(plaintext.data());
        for (int f = 0; f < buffer.getFrameCount(); ++f) {
            for (int ch = 0; ch < buffer.getChannelCount(); ++ch) {
                *floatPtr++ = buffer.getChannelData(ch)[f];
            }
        }

        // Encrypt with AES-256-CTR (streaming: no re-init, context advances)
        std::vector<uint8_t> ciphertext(dataSize);
        int len = 0;

        if (EVP_EncryptUpdate(ctx_, ciphertext.data(), &len,
                              plaintext.data(), static_cast<int>(dataSize)) != 1) {
            std::cerr << "[AES256Encryptor] Encrypt update failed" << std::endl;
            return false;
        }

        // Write encrypted bytes back as floats
        floatPtr = reinterpret_cast<float*>(ciphertext.data());
        for (int f = 0; f < buffer.getFrameCount(); ++f) {
            for (int ch = 0; ch < buffer.getChannelCount(); ++ch) {
                buffer.getChannelData(ch)[f] = *floatPtr++;
            }
        }

        return true;
    }

    PluginInfo getInfo() const override {
        PluginInfo info;
        info.name = "AES-256-CTR Encryptor";
        info.version = "2.1.0";
        info.author = "NDA Team";
        info.description = "AES-256-CTR audio encryption (streaming, frame-size-independent)";
        info.type = PluginType::Processor;
        info.apiVersion = 1;
        return info;
    }

    PluginType getType() const override {
        return PluginType::Processor;
    }

    PluginState getState() const override {
        return state_;
    }

    int getSampleRate() const override {
        return sampleRate_;
    }

    int getChannelCount() const override {
        return channels_;
    }

    void setSampleRate(int rate) override {
        sampleRate_ = rate;
    }

    void setChannelCount(int channels) override {
        channels_ = channels;
    }

    void setParameter(const std::string& key, const std::string& value) override {
        if (key == "key" || key == "aes_256_key") {
            if (value.length() == 64) {  // 32 bytes = 64 hex chars
                for (size_t i = 0; i < 32 && i * 2 < value.length(); ++i) {
                    key_[i] = static_cast<uint8_t>(
                        std::stoi(value.substr(i * 2, 2), nullptr, 16));
                }
                std::cout << "[AES256Encryptor] Key updated from hex" << std::endl;
            } else {
                std::cerr << "[AES256Encryptor] Invalid key format (expected 64 hex chars)" << std::endl;
            }
        }
    }

    std::string getParameter(const std::string& key) const override {
        if (key == "key" || key == "aes_256_key") {
            std::string hex;
            hex.reserve(64);
            for (uint8_t byte : key_) {
                char buf[3];
                snprintf(buf, sizeof(buf), "%02x", byte);
                hex += buf;
            }
            return hex;
        }
        return "";
    }

    double getProcessingLatency() const override {
        return 0.0001;  // ~100 microseconds
    }
};

} // namespace nda

NDA_DECLARE_PLUGIN(nda::AES256EncryptorPlugin)
