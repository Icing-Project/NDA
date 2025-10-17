#ifndef ENCRYPTOR_H
#define ENCRYPTOR_H

#include <vector>
#include <cstdint>
#include <string>

enum class EncryptionAlgorithm {
    AES_128_GCM,
    AES_192_GCM,
    AES_256_GCM,
    CHACHA20_POLY1305
};

class Encryptor
{
public:
    Encryptor(EncryptionAlgorithm algorithm = EncryptionAlgorithm::AES_256_GCM);
    ~Encryptor();

    bool setKey(const std::vector<uint8_t>& key);
    bool generateKey();

    std::vector<uint8_t> getKey() const;

    bool encrypt(const uint8_t* input, size_t inputSize,
                 uint8_t* output, size_t& outputSize);

    bool decrypt(const uint8_t* input, size_t inputSize,
                 uint8_t* output, size_t& outputSize);

    // Real-time audio encryption (optimized)
    bool encryptAudio(float* audioData, size_t sampleCount, int channels);
    bool decryptAudio(float* audioData, size_t sampleCount, int channels);

    bool exportKey(const std::string& filename) const;
    bool importKey(const std::string& filename);

    bool isHardwareAccelerated() const;

private:
    void initializeCipher();
    void generateNonce(uint8_t* nonce, size_t size);

    EncryptionAlgorithm algorithm_;
    std::vector<uint8_t> key_;
    void* cipherContext_;
    bool hardwareAccelerated_;
};

#endif // ENCRYPTOR_H
