#ifndef CRYPTOMANAGER_H
#define CRYPTOMANAGER_H

#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

// Forward declarations for OpenSSL types
typedef struct evp_pkey_st EVP_PKEY;

namespace nda {

/**
 * @brief Centralized cryptographic key management for NDA
 *
 * Provides secure generation, import/export, and storage of cryptographic keys:
 * - AES-256-GCM symmetric keys (32 bytes / 256 bits)
 * - X25519 elliptic curve key pairs for ECDH key exchange
 *
 * Thread-safe singleton for global access from UI and plugin threads.
 * Keys are stored in memory only and securely cleared on shutdown.
 *
 * @thread-safety All methods are thread-safe (mutex-protected)
 */
class CryptoManager {
public:
    /**
     * @brief Get singleton instance
     * @return Reference to the global CryptoManager instance
     */
    static CryptoManager& instance();

    // Prevent copy/move
    CryptoManager(const CryptoManager&) = delete;
    CryptoManager& operator=(const CryptoManager&) = delete;
    CryptoManager(CryptoManager&&) = delete;
    CryptoManager& operator=(CryptoManager&&) = delete;

    ~CryptoManager();

    // ========================
    // AES-256-GCM Symmetric Keys
    // ========================

    /**
     * @brief Generate a new AES-256 key using OpenSSL RAND_bytes
     * @return Hex-encoded key string (64 characters) or empty on error
     */
    std::string generateAES256Key();

    /**
     * @brief Import AES-256 key from hexadecimal string
     * @param hexKey 64-character hex string (32 bytes)
     * @return true if valid and imported, false otherwise
     */
    bool importAES256Key(const std::string& hexKey);

    /**
     * @brief Export current AES-256 key as hexadecimal
     * @return Hex-encoded key (64 characters) or empty if not set
     */
    std::string exportAES256Key() const;

    /**
     * @brief Check if AES-256 key is currently loaded
     * @return true if key is loaded, false otherwise
     */
    bool hasAES256Key() const;

    /**
     * @brief Clear AES-256 key from memory (secure wipe)
     */
    void clearAES256Key();

    // ========================
    // X25519 Key Pair (for ECDH)
    // ========================

    /**
     * @brief Generate a new X25519 key pair using OpenSSL
     * @return true on success, false on error
     */
    bool generateX25519KeyPair();

    /**
     * @brief Import X25519 private key from hex (32 bytes = 64 hex chars)
     * @param hexPrivateKey 64-character hex string
     * @return true if valid and imported, false otherwise
     */
    bool importX25519PrivateKey(const std::string& hexPrivateKey);

    /**
     * @brief Import peer's X25519 public key from hex (32 bytes = 64 hex chars)
     * @param hexPublicKey 64-character hex string
     * @return true if valid and imported, false otherwise
     */
    bool importX25519PeerPublicKey(const std::string& hexPublicKey);

    /**
     * @brief Export local X25519 private key as hexadecimal
     * @return Hex-encoded private key (64 chars) or empty if not set
     */
    std::string exportX25519PrivateKey() const;

    /**
     * @brief Export local X25519 public key as hexadecimal (for sharing)
     * @return Hex-encoded public key (64 chars) or empty if not set
     */
    std::string exportX25519PublicKey() const;

    /**
     * @brief Export peer's X25519 public key as hexadecimal
     * @return Hex-encoded peer public key (64 chars) or empty if not set
     */
    std::string exportX25519PeerPublicKey() const;

    /**
     * @brief Check if X25519 key pair is loaded
     * @return true if keypair is loaded, false otherwise
     */
    bool hasX25519KeyPair() const;

    /**
     * @brief Check if peer's X25519 public key is loaded
     * @return true if peer key is loaded, false otherwise
     */
    bool hasX25519PeerPublicKey() const;

    /**
     * @brief Clear all X25519 keys from memory (secure wipe)
     */
    void clearX25519Keys();

    // ========================
    // ECDH Key Derivation
    // ========================

    /**
     * @brief Perform X25519 ECDH and derive AES-256 key from shared secret
     *
     * Requires: local X25519 key pair and peer's public key must be loaded
     * Result: Shared secret is stored as the active AES-256 key
     *
     * @return true on success (AES key updated), false on error
     */
    bool deriveSharedAES256Key();

    // ========================
    // Error Handling
    // ========================

    /**
     * @brief Get last error message
     * @return Human-readable error string (empty if no error)
     */
    std::string getLastError() const;

private:
    CryptoManager();

    // Helper functions
    bool hexToBytes(const std::string& hex, std::vector<uint8_t>& bytes, size_t expectedSize);
    std::string bytesToHex(const uint8_t* data, size_t size) const;
    void setError(const std::string& error);
    void clearError();
    std::string getOpenSSLError() const;

    // OpenSSL X25519 helpers
    EVP_PKEY* rawBytesToX25519PrivateKey(const std::vector<uint8_t>& rawKey);
    EVP_PKEY* rawBytesToX25519PublicKey(const std::vector<uint8_t>& rawKey);
    bool x25519KeyToRawBytes(EVP_PKEY* key, std::vector<uint8_t>& rawBytes, bool isPrivate);

    // Thread safety
    mutable std::mutex mutex_;

    // AES-256 key storage (32 bytes, securely cleared on destruction)
    std::vector<uint8_t> aes256Key_;
    bool aes256KeyLoaded_;

    // X25519 key storage (EVP_PKEY wrappers)
    EVP_PKEY* x25519PrivateKey_;      // Local private key (contains public key too)
    EVP_PKEY* x25519PeerPublicKey_;   // Peer's public key

    // Error state
    mutable std::string lastError_;
};

} // namespace nda

#endif // CRYPTOMANAGER_H
