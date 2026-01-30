#include "crypto/CryptoManager.h"
#include <openssl/evp.h>
#include <openssl/rand.h>
#include <openssl/err.h>
#include <algorithm>
#include <cctype>
#include <iostream>
#include <iomanip>
#include <sstream>

namespace nda {

// ========================
// Singleton Implementation
// ========================

CryptoManager& CryptoManager::instance() {
    static CryptoManager instance;
    return instance;
}

CryptoManager::CryptoManager()
    : aes256Key_(32, 0)
    , aes256KeyLoaded_(false)
    , x25519PrivateKey_(nullptr)
    , x25519PeerPublicKey_(nullptr)
{
    // OpenSSL is initialized by the application or first use
}

CryptoManager::~CryptoManager() {
    // Securely clear all keys
    clearAES256Key();
    clearX25519Keys();
}

// ========================
// AES-256 Key Management
// ========================

std::string CryptoManager::generateAES256Key() {
    std::lock_guard<std::mutex> lock(mutex_);
    clearError();

    // Generate 32 random bytes using OpenSSL's CSPRNG
    if (RAND_bytes(aes256Key_.data(), 32) != 1) {
        setError("RAND_bytes failed: " + getOpenSSLError());
        return "";
    }

    aes256KeyLoaded_ = true;
    return bytesToHex(aes256Key_.data(), 32);
}

bool CryptoManager::importAES256Key(const std::string& hexKey) {
    std::lock_guard<std::mutex> lock(mutex_);
    clearError();

    std::vector<uint8_t> keyBytes;
    if (!hexToBytes(hexKey, keyBytes, 32)) {
        setError("Invalid hex string: must be 64 hexadecimal characters");
        return false;
    }

    aes256Key_ = keyBytes;
    aes256KeyLoaded_ = true;
    return true;
}

std::string CryptoManager::exportAES256Key() const {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!aes256KeyLoaded_) {
        return "";
    }

    return bytesToHex(aes256Key_.data(), 32);
}

bool CryptoManager::hasAES256Key() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return aes256KeyLoaded_;
}

void CryptoManager::clearAES256Key() {
    std::lock_guard<std::mutex> lock(mutex_);

    // Securely zero memory before clearing
    OPENSSL_cleanse(aes256Key_.data(), aes256Key_.size());
    aes256Key_.assign(32, 0);
    aes256KeyLoaded_ = false;
}

// ========================
// X25519 Key Management
// ========================

bool CryptoManager::generateX25519KeyPair() {
    std::lock_guard<std::mutex> lock(mutex_);
    clearError();

    // Free existing key if present
    if (x25519PrivateKey_) {
        EVP_PKEY_free(x25519PrivateKey_);
        x25519PrivateKey_ = nullptr;
    }

    // Create EVP_PKEY context for X25519
    EVP_PKEY_CTX* ctx = EVP_PKEY_CTX_new_id(EVP_PKEY_X25519, nullptr);
    if (!ctx) {
        setError("Failed to create X25519 context: " + getOpenSSLError());
        return false;
    }

    // Initialize key generation
    if (EVP_PKEY_keygen_init(ctx) <= 0) {
        EVP_PKEY_CTX_free(ctx);
        setError("EVP_PKEY_keygen_init failed: " + getOpenSSLError());
        return false;
    }

    // Generate key pair
    if (EVP_PKEY_keygen(ctx, &x25519PrivateKey_) <= 0) {
        EVP_PKEY_CTX_free(ctx);
        setError("EVP_PKEY_keygen failed: " + getOpenSSLError());
        return false;
    }

    EVP_PKEY_CTX_free(ctx);
    return true;
}

bool CryptoManager::importX25519PrivateKey(const std::string& hexPrivateKey) {
    std::lock_guard<std::mutex> lock(mutex_);
    clearError();

    std::vector<uint8_t> keyBytes;
    if (!hexToBytes(hexPrivateKey, keyBytes, 32)) {
        setError("Invalid hex string: must be 64 hexadecimal characters");
        return false;
    }

    // Free existing key if present
    if (x25519PrivateKey_) {
        EVP_PKEY_free(x25519PrivateKey_);
        x25519PrivateKey_ = nullptr;
    }

    // Convert raw bytes to EVP_PKEY
    x25519PrivateKey_ = rawBytesToX25519PrivateKey(keyBytes);
    if (!x25519PrivateKey_) {
        setError("Failed to import X25519 private key: " + getOpenSSLError());
        return false;
    }

    return true;
}

bool CryptoManager::importX25519PeerPublicKey(const std::string& hexPublicKey) {
    std::lock_guard<std::mutex> lock(mutex_);
    clearError();

    std::vector<uint8_t> keyBytes;
    if (!hexToBytes(hexPublicKey, keyBytes, 32)) {
        setError("Invalid hex string: must be 64 hexadecimal characters");
        return false;
    }

    // Free existing peer key if present
    if (x25519PeerPublicKey_) {
        EVP_PKEY_free(x25519PeerPublicKey_);
        x25519PeerPublicKey_ = nullptr;
    }

    // Convert raw bytes to EVP_PKEY
    x25519PeerPublicKey_ = rawBytesToX25519PublicKey(keyBytes);
    if (!x25519PeerPublicKey_) {
        setError("Failed to import peer X25519 public key: " + getOpenSSLError());
        return false;
    }

    return true;
}

std::string CryptoManager::exportX25519PrivateKey() const {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!x25519PrivateKey_) {
        return "";
    }

    std::vector<uint8_t> rawBytes;
    if (!const_cast<CryptoManager*>(this)->x25519KeyToRawBytes(x25519PrivateKey_, rawBytes, true)) {
        return "";
    }

    return bytesToHex(rawBytes.data(), rawBytes.size());
}

std::string CryptoManager::exportX25519PublicKey() const {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!x25519PrivateKey_) {
        return "";
    }

    std::vector<uint8_t> rawBytes;
    if (!const_cast<CryptoManager*>(this)->x25519KeyToRawBytes(x25519PrivateKey_, rawBytes, false)) {
        return "";
    }

    return bytesToHex(rawBytes.data(), rawBytes.size());
}

std::string CryptoManager::exportX25519PeerPublicKey() const {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!x25519PeerPublicKey_) {
        return "";
    }

    std::vector<uint8_t> rawBytes;
    if (!const_cast<CryptoManager*>(this)->x25519KeyToRawBytes(x25519PeerPublicKey_, rawBytes, false)) {
        return "";
    }

    return bytesToHex(rawBytes.data(), rawBytes.size());
}

std::vector<uint8_t> CryptoManager::exportX25519PrivateKeyBytes() const {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!x25519PrivateKey_) {
        return {};
    }

    std::vector<uint8_t> rawBytes;
    if (!const_cast<CryptoManager*>(this)->x25519KeyToRawBytes(x25519PrivateKey_, rawBytes, true)) {
        return {};
    }

    return rawBytes;
}

std::vector<uint8_t> CryptoManager::exportX25519PublicKeyBytes() const {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!x25519PrivateKey_) {
        return {};
    }

    std::vector<uint8_t> rawBytes;
    if (!const_cast<CryptoManager*>(this)->x25519KeyToRawBytes(x25519PrivateKey_, rawBytes, false)) {
        return {};
    }

    return rawBytes;
}

std::vector<uint8_t> CryptoManager::exportX25519PeerPublicKeyBytes() const {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!x25519PeerPublicKey_) {
        return {};
    }

    std::vector<uint8_t> rawBytes;
    if (!const_cast<CryptoManager*>(this)->x25519KeyToRawBytes(x25519PeerPublicKey_, rawBytes, false)) {
        return {};
    }

    return rawBytes;
}

bool CryptoManager::hasX25519KeyPair() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return x25519PrivateKey_ != nullptr;
}

bool CryptoManager::hasX25519PeerPublicKey() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return x25519PeerPublicKey_ != nullptr;
}

void CryptoManager::clearX25519Keys() {
    std::lock_guard<std::mutex> lock(mutex_);

    if (x25519PrivateKey_) {
        EVP_PKEY_free(x25519PrivateKey_);  // OpenSSL handles secure cleanup
        x25519PrivateKey_ = nullptr;
    }

    if (x25519PeerPublicKey_) {
        EVP_PKEY_free(x25519PeerPublicKey_);
        x25519PeerPublicKey_ = nullptr;
    }
}

// ========================
// ECDH Key Derivation
// ========================

bool CryptoManager::deriveSharedAES256Key() {
    std::lock_guard<std::mutex> lock(mutex_);
    clearError();

    if (!x25519PrivateKey_) {
        setError("No X25519 key pair loaded");
        return false;
    }

    if (!x25519PeerPublicKey_) {
        setError("No peer X25519 public key loaded");
        return false;
    }

    // Create derivation context
    EVP_PKEY_CTX* ctx = EVP_PKEY_CTX_new(x25519PrivateKey_, nullptr);
    if (!ctx) {
        setError("Failed to create ECDH context: " + getOpenSSLError());
        return false;
    }

    // Initialize ECDH
    if (EVP_PKEY_derive_init(ctx) <= 0) {
        EVP_PKEY_CTX_free(ctx);
        setError("EVP_PKEY_derive_init failed: " + getOpenSSLError());
        return false;
    }

    // Set peer key
    if (EVP_PKEY_derive_set_peer(ctx, x25519PeerPublicKey_) <= 0) {
        EVP_PKEY_CTX_free(ctx);
        setError("EVP_PKEY_derive_set_peer failed: " + getOpenSSLError());
        return false;
    }

    // Determine shared secret length
    size_t sharedSecretLen = 0;
    if (EVP_PKEY_derive(ctx, nullptr, &sharedSecretLen) <= 0) {
        EVP_PKEY_CTX_free(ctx);
        setError("Failed to determine shared secret length: " + getOpenSSLError());
        return false;
    }

    // Derive shared secret (X25519 produces 32 bytes)
    std::vector<uint8_t> sharedSecret(sharedSecretLen);
    if (EVP_PKEY_derive(ctx, sharedSecret.data(), &sharedSecretLen) <= 0) {
        EVP_PKEY_CTX_free(ctx);
        setError("EVP_PKEY_derive failed: " + getOpenSSLError());
        return false;
    }

    EVP_PKEY_CTX_free(ctx);

    // Use shared secret directly as AES-256 key
    // Note: In production, apply HKDF for proper key derivation
    if (sharedSecret.size() != 32) {
        setError("Unexpected shared secret length: " + std::to_string(sharedSecret.size()));
        return false;
    }

    aes256Key_ = sharedSecret;
    aes256KeyLoaded_ = true;

    return true;
}

// ========================
// Helper Functions
// ========================

bool CryptoManager::hexToBytes(const std::string& hex, std::vector<uint8_t>& bytes, size_t expectedSize) {
    // Check length
    if (hex.length() != expectedSize * 2) {
        return false;
    }

    // Validate all characters are hex digits
    if (!std::all_of(hex.begin(), hex.end(), [](char c) { return std::isxdigit(c); })) {
        return false;
    }

    // Convert hex string to bytes
    bytes.resize(expectedSize);
    for (size_t i = 0; i < expectedSize; ++i) {
        std::string byteStr = hex.substr(i * 2, 2);
        try {
            bytes[i] = static_cast<uint8_t>(std::stoi(byteStr, nullptr, 16));
        } catch (const std::exception&) {
            return false;
        }
    }

    return true;
}

std::string CryptoManager::bytesToHex(const uint8_t* data, size_t size) const {
    std::ostringstream oss;
    oss << std::hex << std::setfill('0');

    for (size_t i = 0; i < size; ++i) {
        oss << std::setw(2) << static_cast<int>(data[i]);
    }

    return oss.str();
}

void CryptoManager::setError(const std::string& error) {
    lastError_ = error;
    std::cerr << "[CryptoManager] Error: " << error << std::endl;
}

void CryptoManager::clearError() {
    lastError_.clear();
}

std::string CryptoManager::getOpenSSLError() const {
    unsigned long err = ERR_get_error();
    if (err == 0) {
        return "Unknown OpenSSL error";
    }

    char buf[256];
    ERR_error_string_n(err, buf, sizeof(buf));
    return std::string(buf);
}

std::string CryptoManager::getLastError() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return lastError_;
}

// ========================
// OpenSSL X25519 Helpers
// ========================

EVP_PKEY* CryptoManager::rawBytesToX25519PrivateKey(const std::vector<uint8_t>& rawKey) {
    if (rawKey.size() != 32) {
        setError("X25519 private key must be 32 bytes");
        return nullptr;
    }

    // Create EVP_PKEY from raw private key bytes
    EVP_PKEY* pkey = EVP_PKEY_new_raw_private_key(EVP_PKEY_X25519, nullptr, rawKey.data(), rawKey.size());
    if (!pkey) {
        setError("EVP_PKEY_new_raw_private_key failed: " + getOpenSSLError());
        return nullptr;
    }

    return pkey;
}

EVP_PKEY* CryptoManager::rawBytesToX25519PublicKey(const std::vector<uint8_t>& rawKey) {
    if (rawKey.size() != 32) {
        setError("X25519 public key must be 32 bytes");
        return nullptr;
    }

    // Create EVP_PKEY from raw public key bytes
    EVP_PKEY* pkey = EVP_PKEY_new_raw_public_key(EVP_PKEY_X25519, nullptr, rawKey.data(), rawKey.size());
    if (!pkey) {
        setError("EVP_PKEY_new_raw_public_key failed: " + getOpenSSLError());
        return nullptr;
    }

    return pkey;
}

bool CryptoManager::x25519KeyToRawBytes(EVP_PKEY* key, std::vector<uint8_t>& rawBytes, bool isPrivate) {
    if (!key) {
        setError("Null EVP_PKEY");
        return false;
    }

    size_t keyLen = 0;

    if (isPrivate) {
        // Get private key length
        if (EVP_PKEY_get_raw_private_key(key, nullptr, &keyLen) != 1) {
            setError("Failed to get private key length: " + getOpenSSLError());
            return false;
        }

        rawBytes.resize(keyLen);

        // Extract private key
        if (EVP_PKEY_get_raw_private_key(key, rawBytes.data(), &keyLen) != 1) {
            setError("Failed to extract private key: " + getOpenSSLError());
            return false;
        }
    } else {
        // Get public key length
        if (EVP_PKEY_get_raw_public_key(key, nullptr, &keyLen) != 1) {
            setError("Failed to get public key length: " + getOpenSSLError());
            return false;
        }

        rawBytes.resize(keyLen);

        // Extract public key
        if (EVP_PKEY_get_raw_public_key(key, rawBytes.data(), &keyLen) != 1) {
            setError("Failed to extract public key: " + getOpenSSLError());
            return false;
        }
    }

    return true;
}

} // namespace nda
