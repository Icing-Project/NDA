#ifndef DASHBOARD_H
#define DASHBOARD_H

#include <QWidget>
#include <QPushButton>
#include <QLabel>
#include <QProgressBar>
#include <QGroupBox>
#include "core/ProcessingPipeline.h"
#include <memory>

class Dashboard : public QWidget
{
    Q_OBJECT

public:
    explicit Dashboard(QWidget *parent = nullptr);
    ~Dashboard();

    void setPipeline(std::shared_ptr<nda::ProcessingPipeline> pipeline);
    void updatePipelineState();

signals:
    void streamStarted();
    void streamStopped();

private slots:
    void onStartStopClicked();
    void updateAudioMeters();

private:
    void setupUI();
    void applyModernStyles();
    QWidget* createCard();
    QWidget* createMetricCard();
    QProgressBar* createModernProgressBar();
    void updateMeterColor(QProgressBar *meter, int value);

    QPushButton *startStopButton;
    QLabel *statusLabel;
    QLabel *latencyLabel;
    QLabel *cpuLabel;
    QLabel *memoryLabel;

    // Audio meters
    QProgressBar *inputMeterL;
    QProgressBar *inputMeterR;
    QProgressBar *outputMeterL;
    QProgressBar *outputMeterR;

    bool isStreaming;

    // Pipeline reference
    std::shared_ptr<nda::ProcessingPipeline> pipeline_;
};

#endif // DASHBOARD_H
