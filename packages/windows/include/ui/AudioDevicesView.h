#ifndef AUDIODEVICESVIEW_H
#define AUDIODEVICESVIEW_H

#include <QWidget>
#include <QListWidget>
#include <QComboBox>
#include <QPushButton>

class AudioDevicesView : public QWidget
{
    Q_OBJECT

public:
    explicit AudioDevicesView(QWidget *parent = nullptr);
    ~AudioDevicesView();

private slots:
    void onRefreshDevices();
    void onInputDeviceChanged(int index);
    void onOutputDeviceChanged(int index);

private:
    void setupUI();
    void populateDevices();

    QComboBox *inputDeviceCombo;
    QComboBox *outputDeviceCombo;
    QComboBox *sampleRateCombo;
    QComboBox *bufferSizeCombo;
    QPushButton *refreshButton;
    QListWidget *deviceInfoList;
};

#endif // AUDIODEVICESVIEW_H
