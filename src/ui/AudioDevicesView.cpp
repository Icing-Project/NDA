#include "ui/AudioDevicesView.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGroupBox>
#include <QLabel>

AudioDevicesView::AudioDevicesView(QWidget *parent)
    : QWidget(parent)
{
    setupUI();
    populateDevices();
}

AudioDevicesView::~AudioDevicesView()
{
}

void AudioDevicesView::setupUI()
{
    QVBoxLayout *mainLayout = new QVBoxLayout(this);

    // Input device selection
    QGroupBox *inputGroup = new QGroupBox("Input Device", this);
    QVBoxLayout *inputLayout = new QVBoxLayout(inputGroup);

    inputDeviceCombo = new QComboBox(this);
    connect(inputDeviceCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &AudioDevicesView::onInputDeviceChanged);
    inputLayout->addWidget(inputDeviceCombo);

    mainLayout->addWidget(inputGroup);

    // Output device selection
    QGroupBox *outputGroup = new QGroupBox("Output Device", this);
    QVBoxLayout *outputLayout = new QVBoxLayout(outputGroup);

    outputDeviceCombo = new QComboBox(this);
    connect(outputDeviceCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &AudioDevicesView::onOutputDeviceChanged);
    outputLayout->addWidget(outputDeviceCombo);

    mainLayout->addWidget(outputGroup);

    // Audio settings
    QGroupBox *settingsGroup = new QGroupBox("Audio Settings", this);
    QVBoxLayout *settingsLayout = new QVBoxLayout(settingsGroup);

    QHBoxLayout *sampleRateLayout = new QHBoxLayout();
    sampleRateLayout->addWidget(new QLabel("Sample Rate:", this));
    sampleRateCombo = new QComboBox(this);
    sampleRateCombo->addItems({"44100 Hz", "48000 Hz", "96000 Hz", "192000 Hz"});
    sampleRateCombo->setCurrentIndex(1); // Default to 48kHz
    sampleRateLayout->addWidget(sampleRateCombo);
    settingsLayout->addLayout(sampleRateLayout);

    QHBoxLayout *bufferSizeLayout = new QHBoxLayout();
    bufferSizeLayout->addWidget(new QLabel("Buffer Size:", this));
    bufferSizeCombo = new QComboBox(this);
    bufferSizeCombo->addItems({"64 samples", "128 samples", "256 samples", "512 samples", "1024 samples"});
    bufferSizeCombo->setCurrentIndex(1); // Default to 128
    bufferSizeLayout->addWidget(bufferSizeCombo);
    settingsLayout->addLayout(bufferSizeLayout);

    mainLayout->addWidget(settingsGroup);

    // Refresh button
    refreshButton = new QPushButton("Refresh Devices", this);
    connect(refreshButton, &QPushButton::clicked, this, &AudioDevicesView::onRefreshDevices);
    mainLayout->addWidget(refreshButton);

    // Device info
    QGroupBox *infoGroup = new QGroupBox("Device Information", this);
    QVBoxLayout *infoLayout = new QVBoxLayout(infoGroup);

    deviceInfoList = new QListWidget(this);
    infoLayout->addWidget(deviceInfoList);

    mainLayout->addWidget(infoGroup);

    mainLayout->addStretch();
}

void AudioDevicesView::populateDevices()
{
    // Placeholder - will be populated with real devices
    inputDeviceCombo->clear();
    outputDeviceCombo->clear();

    inputDeviceCombo->addItem("WASAPI: Default Input");
    inputDeviceCombo->addItem("WASAPI: Microphone (Realtek)");
    inputDeviceCombo->addItem("ASIO: Focusrite USB");

    outputDeviceCombo->addItem("WASAPI: Default Output");
    outputDeviceCombo->addItem("WASAPI: Speakers (Realtek)");
    outputDeviceCombo->addItem("ASIO: Focusrite USB");

    deviceInfoList->clear();
    deviceInfoList->addItem("API: WASAPI");
    deviceInfoList->addItem("Channels: 2 (Stereo)");
    deviceInfoList->addItem("Sample Rate: 48000 Hz");
    deviceInfoList->addItem("Latency: ~3-5 ms");
}

void AudioDevicesView::onRefreshDevices()
{
    populateDevices();
}

void AudioDevicesView::onInputDeviceChanged(int index)
{
    // Placeholder - will update device info
}

void AudioDevicesView::onOutputDeviceChanged(int index)
{
    // Placeholder - will update device info
}
