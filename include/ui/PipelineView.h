#ifndef PIPELINEVIEW_H
#define PIPELINEVIEW_H

#include <QWidget>
#include <QComboBox>
#include <QPushButton>
#include <QLabel>
#include <QGroupBox>
#include "core/ProcessingPipeline.h"
#include "plugins/PluginManager.h"
#include <memory>

class PipelineView : public QWidget
{
    Q_OBJECT

public:
    explicit PipelineView(QWidget *parent = nullptr);
    ~PipelineView();

    void setPluginManager(std::shared_ptr<nda::PluginManager> manager);
    void setPipeline(std::shared_ptr<nda::ProcessingPipeline> pipeline);

signals:
    void pipelineStarted();
    void pipelineStopped();
    void pipelineConfigured();

private slots:
    void onLoadPluginsClicked();
    void onAutoLoadPythonPlugins();
    void onStartPipelineClicked();
    void onStopPipelineClicked();
    void onAudioSourceChanged(int index);
    void onBearerChanged(int index);
    void onEncryptorChanged(int index);
    void onAudioSinkChanged(int index);

private:
    void setupUI();
    void refreshPluginLists();
    void updatePipelineStatus();
    QWidget* createStepCard();
    void applyModernStyles();

    // Plugin selection
    QComboBox *audioSourceCombo;
    QComboBox *bearerCombo;
    QComboBox *encryptorCombo;
    QComboBox *audioSinkCombo;

    // Controls
    QPushButton *loadPluginsButton;
    QPushButton *autoLoadPythonButton;
    QPushButton *startPipelineButton;
    QPushButton *stopPipelineButton;

    // Status
    QLabel *pipelineStatusLabel;
    QLabel *latencyLabel;
    QLabel *throughputLabel;

    // Pipeline components
    std::shared_ptr<nda::PluginManager> pluginManager_;
    std::shared_ptr<nda::ProcessingPipeline> pipeline_;

    // Selected plugins
    std::shared_ptr<nda::AudioSourcePlugin> selectedSource_;
    std::shared_ptr<nda::BearerPlugin> selectedBearer_;
    std::shared_ptr<nda::EncryptorPlugin> selectedEncryptor_;
    std::shared_ptr<nda::AudioSinkPlugin> selectedSink_;
};

#endif // PIPELINEVIEW_H
