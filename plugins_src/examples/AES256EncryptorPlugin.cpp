/**
 * AES-256-CTR Encryptor Processor Plugin
 *
 * Symmetric encryption using OpenSSL EVP API with AES-256 in CTR mode.
 * Uses a deterministic frame counter for the IV so that the matching
 * decryptor can reproduce it without out-of-band nonce transmission.
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
    uint64_t frameCounter_;        // Deterministic IV counter

public:
    AES256EncryptorPlugin()
        : ctx_(nullptr)
        , key_(32, 0)
        , sampleRate_(48000)
        , channels_(2)
        , state_(PluginState::Unloaded)
        , frameCounter_(0)
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
        state_ = PluginState::Running;
        frameCounter_ = 0;
        std::cout << "[AES256Encryptor] Started" << std::endl;
        return true;
    }

    void stop() override {
        if (state_ == PluginState::Running) {
            state_ = PluginState::Initialized;
            std::cout << "[AES256Encryptor] Stopped after " << frameCounter_ << " frames" << std::endl;
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

        // Build deterministic 16-byte IV from frame counter
        // IV = 8 zero bytes + 8-byte big-endian counter
        uint8_t iv[16] = {0};
        uint64_t counter = frameCounter_;
        for (int i = 15; i >= 8; --i) {
            iv[i] = static_cast<uint8_t>(counter & 0xFF);
            counter >>= 8;
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

        // Encrypt with AES-256-CTR (ciphertext size == plaintext size)
        std::vector<uint8_t> ciphertext(dataSize);
        int len = 0;

        if (EVP_EncryptInit_ex(ctx_, EVP_aes_256_ctr(), nullptr,
                               key_.data(), iv) != 1) {
            std::cerr << "[AES256Encryptor] Encrypt init failed" << std::endl;
            return false;
        }

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

        frameCounter_++;
        return true;
    }

    PluginInfo getInfo() const override {
        PluginInfo info;
        info.name = "AES-256-CTR Encryptor";
        info.version = "2.0.0";
        info.author = "NDA Team";
        info.description = "AES-256-CTR audio encryption with deterministic counter IV";
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
