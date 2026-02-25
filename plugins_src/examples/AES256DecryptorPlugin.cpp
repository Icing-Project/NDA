/**
 * AES-256-CTR Decryptor Processor Plugin
 *
 * Symmetric decryption using OpenSSL EVP API with AES-256 in CTR mode.
 * The cipher context is initialized once in start() with an all-zero IV and
 * then used as a streaming cipher across processAudio() calls. This makes the
 * decryption byte-position-based rather than per-buffer, so it works correctly
 * regardless of the buffer size the pipeline negotiates — including when the
 * encryptor and decryptor pipelines happen to negotiate different frame sizes.
 *
 * WARNING: Key exchange is out of scope. Keys must be shared out-of-band
 * or via the NDA Crypto menu (Generate AES Key -> Apply to Pipeline).
 * Must use the same key as the AES256EncryptorPlugin.
 */

#include "plugins/AudioProcessorPlugin.h"
#include "plugins/PluginTypes.h"
#include <openssl/evp.h>
#include <vector>
#include <cstring>
#include <iostream>

namespace nda {

class AES256DecryptorPlugin : public AudioProcessorPlugin {
private:
    EVP_CIPHER_CTX* ctx_;
    std::vector<uint8_t> key_;     // 256-bit decryption key
    int sampleRate_;
    int channels_;
    PluginState state_;

public:
    AES256DecryptorPlugin()
        : ctx_(nullptr)
        , key_(32, 0)
        , sampleRate_(48000)
        , channels_(2)
        , state_(PluginState::Unloaded)
    {
        ctx_ = EVP_CIPHER_CTX_new();
    }

    ~AES256DecryptorPlugin() {
        if (ctx_) {
            EVP_CIPHER_CTX_free(ctx_);
            ctx_ = nullptr;
        }
    }

    bool initialize() override {
        if (!ctx_) {
            std::cerr << "[AES256Decryptor] Failed to create cipher context" << std::endl;
            state_ = PluginState::Error;
            return false;
        }

        // Key must be set via setParameter or Crypto menu before start
        state_ = PluginState::Initialized;
        std::cout << "[AES256Decryptor] Initialized (waiting for key via Crypto menu or setParameter)" << std::endl;
        return true;
    }

    bool start() override {
        if (state_ != PluginState::Initialized) {
            return false;
        }

        // Verify key is set (not all zeros)
        bool keySet = false;
        for (uint8_t byte : key_) {
            if (byte != 0) {
                keySet = true;
                break;
            }
        }

        if (!keySet) {
            std::cerr << "[AES256Decryptor] Error: Key not set. Use Crypto menu -> Generate AES Key -> Apply to Pipeline" << std::endl;
            return false;
        }

        // Initialize streaming CTR context once with all-zero IV.
        // All subsequent processAudio() calls use EVP_DecryptUpdate only,
        // maintaining continuous keystream position regardless of buffer size.
        uint8_t iv[16] = {0};
        if (EVP_DecryptInit_ex(ctx_, EVP_aes_256_ctr(), nullptr,
                               key_.data(), iv) != 1) {
            std::cerr << "[AES256Decryptor] Failed to initialize CTR context" << std::endl;
            state_ = PluginState::Error;
            return false;
        }

        state_ = PluginState::Running;
        std::cout << "[AES256Decryptor] Started (streaming CTR, IV=0)" << std::endl;
        return true;
    }

    void stop() override {
        if (state_ == PluginState::Running) {
            state_ = PluginState::Initialized;
            std::cout << "[AES256Decryptor] Stopped" << std::endl;
        }
    }

    void shutdown() override {
        state_ = PluginState::Unloaded;
        key_.assign(32, 0);  // Clear key from memory
        std::cout << "[AES256Decryptor] Shutdown" << std::endl;
    }

    bool processAudio(AudioBuffer& buffer) override {
        if (state_ != PluginState::Running || !ctx_) {
            return false;
        }

        // Convert encrypted float data to bytes
        size_t totalSamples = static_cast<size_t>(buffer.getFrameCount()) * buffer.getChannelCount();
        size_t dataSize = totalSamples * sizeof(float);
        std::vector<uint8_t> ciphertext(dataSize);

        float* floatPtr = reinterpret_cast<float*>(ciphertext.data());
        for (int f = 0; f < buffer.getFrameCount(); ++f) {
            for (int ch = 0; ch < buffer.getChannelCount(); ++ch) {
                *floatPtr++ = buffer.getChannelData(ch)[f];
            }
        }

        // Decrypt with AES-256-CTR (streaming: no re-init, context advances)
        std::vector<uint8_t> plaintext(dataSize);
        int len = 0;

        if (EVP_DecryptUpdate(ctx_, plaintext.data(), &len,
                              ciphertext.data(), static_cast<int>(dataSize)) != 1) {
            std::cerr << "[AES256Decryptor] Decrypt update failed" << std::endl;
            return false;
        }

        // Convert decrypted bytes back to float buffer
        floatPtr = reinterpret_cast<float*>(plaintext.data());
        for (int f = 0; f < buffer.getFrameCount(); ++f) {
            for (int ch = 0; ch < buffer.getChannelCount(); ++ch) {
                buffer.getChannelData(ch)[f] = *floatPtr++;
            }
        }

        return true;
    }

    PluginInfo getInfo() const override {
        PluginInfo info;
        info.name = "AES-256-CTR Decryptor";
        info.version = "2.1.0";
        info.author = "NDA Team";
        info.description = "AES-256-CTR audio decryption (streaming, frame-size-independent)";
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
                    try {
                        key_[i] = static_cast<uint8_t>(
                            std::stoi(value.substr(i * 2, 2), nullptr, 16));
                    } catch (...) {
                        std::cerr << "[AES256Decryptor] Invalid hex in key" << std::endl;
                        return;
                    }
                }
                std::cout << "[AES256Decryptor] Key updated from hex" << std::endl;
            } else {
                std::cerr << "[AES256Decryptor] Invalid key format (expected 64 hex chars)" << std::endl;
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

NDA_DECLARE_PLUGIN(nda::AES256DecryptorPlugin)
