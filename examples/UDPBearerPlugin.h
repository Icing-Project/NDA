#ifndef UDPBEARERPLUGIN_H
#define UDPBEARERPLUGIN_H

#include "plugins/BearerPlugin.h"
#include <string>

namespace NADE {

class UDPBearerPlugin : public BearerPlugin {
public:
    UDPBearerPlugin();
    ~UDPBearerPlugin() override;

    // BasePlugin interface
    bool initialize() override;
    void shutdown() override;
    bool start() override;
    void stop() override;
    PluginInfo getInfo() const override;
    PluginState getState() const override;
    void setParameter(const std::string& key, const std::string& value) override;
    std::string getParameter(const std::string& key) const override;

    // BearerPlugin interface
    bool connect(const std::string& address, int port) override;
    void disconnect() override;
    bool isConnected() const override { return connected_; }

    bool sendPacket(const Packet& packet) override;
    void setPacketReceivedCallback(PacketReceivedCallback callback) override;

    void setLatency(int milliseconds) override { simulatedLatency_ = milliseconds; }
    void setPacketLoss(float percentage) override { packetLoss_ = percentage; }
    void setJitter(int milliseconds) override { jitter_ = milliseconds; }
    void setBandwidth(int bitsPerSecond) override { bandwidth_ = bitsPerSecond; }

    int getLatency() const override { return simulatedLatency_; }
    float getPacketLossRate() const override { return actualPacketLoss_; }
    uint64_t getBytesSent() const override { return bytesSent_; }
    uint64_t getBytesReceived() const override { return bytesReceived_; }

private:
    void receiveThread();
    bool shouldDropPacket();

    PluginState state_;
    bool connected_;
    std::string remoteAddress_;
    int remotePort_;

    int socketFd_;
    PacketReceivedCallback receiveCallback_;

    // Network simulation parameters
    int simulatedLatency_;
    float packetLoss_;
    int jitter_;
    int bandwidth_;

    // Statistics
    float actualPacketLoss_;
    uint64_t bytesSent_;
    uint64_t bytesReceived_;
    uint64_t packetsSent_;
    uint64_t packetsDropped_;
};

} // namespace NADE

NADE_DECLARE_PLUGIN(NADE::UDPBearerPlugin)

#endif // UDPBEARERPLUGIN_H
