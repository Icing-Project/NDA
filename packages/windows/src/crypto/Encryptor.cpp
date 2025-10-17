#include "crypto/Encryptor.h"
#include <fstream>
#include <cstring>
#include <random>

Encryptor::Encryptor(EncryptionAlgorithm algorithm)
    : algorithm_(algorithm), cipherContext_(nullptr), hardwareAccelerated_(false)
{
    initializeCipher();
}

Encryptor::~Encryptor()
{
    // Cleanup OpenSSL cipher context
}

void Encryptor::initializeCipher()
{
    // Placeholder - will initialize OpenSSL cipher context
    // Check for AES-NI support
    hardwareAccelerated_ = true; // Assume hardware acceleration for now
}

bool Encryptor::setKey(const std::vector<uint8_t>& key)
{
    int expectedKeySize = 32; // 256-bit for AES-256
    if (algorithm_ == EncryptionAlgorithm::AES_128_GCM) {
        expectedKeySize = 16;
    } else if (algorithm_ == EncryptionAlgorithm::AES_192_GCM) {
        expectedKeySize = 24;
    }

    if (key.size() != static_cast<size_t>(expectedKeySize)) {
        return false;
    }

    key_ = key;
    return true;
}

bool Encryptor::generateKey()
{
    int keySize = 32; // Default to 256-bit
    if (algorithm_ == EncryptionAlgorithm::AES_128_GCM) {
        keySize = 16;
    } else if (algorithm_ == EncryptionAlgorithm::AES_192_GCM) {
        keySize = 24;
    }

    key_.resize(keySize);

    // Generate random key
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 255);

    for (int i = 0; i < keySize; ++i) {
        key_[i] = static_cast<uint8_t>(dis(gen));
    }

    return true;
}

std::vector<uint8_t> Encryptor::getKey() const
{
    return key_;
}

bool Encryptor::encrypt(const uint8_t* input, size_t inputSize,
                        uint8_t* output, size_t& outputSize)
{
    if (key_.empty()) {
        return false;
    }

    // Placeholder - will use OpenSSL EVP API for encryption
    // For now, just copy data (no actual encryption)
    if (outputSize < inputSize + 16) { // +16 for GCM tag
        return false;
    }

    std::memcpy(output, input, inputSize);
    outputSize = inputSize + 16;

    return true;
}

bool Encryptor::decrypt(const uint8_t* input, size_t inputSize,
                        uint8_t* output, size_t& outputSize)
{
    if (key_.empty()) {
        return false;
    }

    // Placeholder - will use OpenSSL EVP API for decryption
    if (inputSize < 16) { // Need at least GCM tag
        return false;
    }

    std::memcpy(output, input, inputSize - 16);
    outputSize = inputSize - 16;

    return true;
}

bool Encryptor::encryptAudio(float* audioData, size_t sampleCount, int channels)
{
    if (key_.empty()) {
        return false;
    }

    // Placeholder - optimized audio encryption
    // Will use SIMD and hardware acceleration
    size_t totalSamples = sampleCount * channels;
    uint8_t* byteData = reinterpret_cast<uint8_t*>(audioData);
    size_t byteSize = totalSamples * sizeof(float);

    size_t outputSize = byteSize + 16;
    return encrypt(byteData, byteSize, byteData, outputSize);
}

bool Encryptor::decryptAudio(float* audioData, size_t sampleCount, int channels)
{
    if (key_.empty()) {
        return false;
    }

    // Placeholder - optimized audio decryption
    size_t totalSamples = sampleCount * channels;
    uint8_t* byteData = reinterpret_cast<uint8_t*>(audioData);
    size_t byteSize = totalSamples * sizeof(float) + 16;

    size_t outputSize = byteSize;
    return decrypt(byteData, byteSize, byteData, outputSize);
}

bool Encryptor::exportKey(const std::string& filename) const
{
    if (key_.empty()) {
        return false;
    }

    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }

    file.write(reinterpret_cast<const char*>(key_.data()), key_.size());
    return true;
}

bool Encryptor::importKey(const std::string& filename)
{
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        return false;
    }

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    key_.resize(size);
    if (!file.read(reinterpret_cast<char*>(key_.data()), size)) {
        return false;
    }

    return true;
}

void Encryptor::generateNonce(uint8_t* nonce, size_t size)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 255);

    for (size_t i = 0; i < size; ++i) {
        nonce[i] = static_cast<uint8_t>(dis(gen));
    }
}

bool Encryptor::isHardwareAccelerated() const
{
    return hardwareAccelerated_;
}
