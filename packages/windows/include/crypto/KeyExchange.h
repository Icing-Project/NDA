#ifndef KEYEXCHANGE_H
#define KEYEXCHANGE_H

#include <string>
#include <vector>
#include <cstdint>

enum class KeyExchangeMethod {
    ECDH_P256,    // Elliptic Curve Diffie-Hellman with P-256 curve
    ECDH_P384,    // Elliptic Curve Diffie-Hellman with P-384 curve
    DH_2048,      // Diffie-Hellman 2048-bit
    DH_4096       // Diffie-Hellman 4096-bit
};

class KeyExchange
{
public:
    KeyExchange(KeyExchangeMethod method = KeyExchangeMethod::ECDH_P256);
    ~KeyExchange();

    bool generateKeyPair();
    std::vector<uint8_t> getPublicKey() const;
    bool setRemotePublicKey(const std::vector<uint8_t>& publicKey);

    std::vector<uint8_t> deriveSharedSecret();

    bool exportPublicKey(const std::string& filename) const;
    bool importRemotePublicKey(const std::string& filename);

private:
    KeyExchangeMethod method_;
    void* keyPair_;
    std::vector<uint8_t> publicKey_;
    std::vector<uint8_t> remotePublicKey_;
    std::vector<uint8_t> sharedSecret_;
};

#endif // KEYEXCHANGE_H
