#ifndef BEARERPLUGIN_H
#define BEARERPLUGIN_H

#include "BasePlugin.h"
#include <vector>
#include <cstdint>
#include <functional>

namespace NADE {

// Network packet structure
struct Packet {
    std::vector<uint8_t> data;
    uint64_t timestamp;
    uint32_t sequenceNumber;
    uint16_t flags;
};

// Callback for received packets
using PacketReceivedCallback = std::function<void(const Packet& packet)>;

class BearerPlugin : public BasePlugin {
public:
    virtual ~BearerPlugin() = default;

    PluginType getType() const override { return PluginType::Bearer; }

    // Bearer specific methods
    virtual bool connect(const std::string& address, int port) = 0;
    virtual void disconnect() = 0;
    virtual bool isConnected() const = 0;

    // Send/Receive
    virtual bool sendPacket(const Packet& packet) = 0;
    virtual void setPacketReceivedCallback(PacketReceivedCallback callback) = 0;

    // Network simulation
    virtual void setLatency(int milliseconds) = 0;
    virtual void setPacketLoss(float percentage) = 0;
    virtual void setJitter(int milliseconds) = 0;
    virtual void setBandwidth(int bitsPerSecond) = 0;

    // Statistics
    virtual int getLatency() const = 0;
    virtual float getPacketLossRate() const = 0;
    virtual uint64_t getBytesSent() const = 0;
    virtual uint64_t getBytesReceived() const = 0;
};

} // namespace NADE

#endif // BEARERPLUGIN_H
