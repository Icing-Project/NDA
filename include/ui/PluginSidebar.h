#ifndef PLUGINSIDEBAR_H
#define PLUGINSIDEBAR_H

#include <QWidget>
#include <QVBoxLayout>
#include <QLabel>
#include <QComboBox>
#include <QSlider>
#include <QLineEdit>
#include <QPushButton>
#include <QCheckBox>
#include <QFileDialog>
#include <QPointer>
#include "plugins/BasePlugin.h"
#include <memory>
#include <map>
#include <mutex>

namespace nda {

/**
 * Plugin-specific parameter configuration sidebar
 * 
 * Dynamically generates UI based on plugin's available parameters.
 * Supports:
 * - Audio device selection (QComboBox)
 * - File path selection (QLineEdit + Browse button)
 * - Numeric parameters (QSlider + SpinBox)
 * - Boolean flags (QCheckBox)
 * - String parameters (QLineEdit)
 */
class PluginSidebar : public QWidget
{
    Q_OBJECT

public:
    explicit PluginSidebar(QWidget *parent = nullptr);
    ~PluginSidebar();

    void showPluginConfig(std::shared_ptr<BasePlugin> plugin);

signals:
    void aboutToApplyParameters();  // Emitted before parameters are applied
    void parameterChanged(const std::string& key, const std::string& value);

private slots:
    void onApplyClicked();
    void onResetClicked();

private:
    void setupUI();
    void clearParameters();
    void addDeviceSelector(const QString& label, const QString& key);
    void addFileSelector(const QString& label, const QString& key);
    void addSlider(const QString& label, const QString& key, int min, int max, int defaultVal);
    void addCheckbox(const QString& label, const QString& key);
    void addTextInput(const QString& label, const QString& key);

    // v2.2: AIOC-specific UI helpers
    void addAIOCDeviceSelector(const QString& label, const QString& key, int direction);
    void addPTTModeSelector(const QString& label, const QString& key);

    // v2.3: Linux PulseAudio device selector
    void addPulseDeviceSelector(const QString& label, const QString& key, bool isSource);

    // v2.3: Windows WASAPI device selector (generic, for Microphone/Speaker plugins)
    void addWASAPIDeviceSelector(const QString& label, const QString& key, int direction);

    // v2.4: Linux AIOC PTT mode selector (HID/CDC/Auto)
    void addLinuxPTTModeSelector(const QString& label, const QString& key);

    void applyModernStyles();

    // RAII guard for exception-safe isUpdating_ management
    class UpdateGuard {
    public:
        UpdateGuard(bool& flag) : flag_(flag) { flag_ = true; }
        ~UpdateGuard() { flag_ = false; }  // Always resets, even on exception
    private:
        bool& flag_;
    };

    std::shared_ptr<BasePlugin> currentPlugin_;

    QVBoxLayout *mainLayout_;
    QWidget *contentWidget_;
    QLabel *pluginNameLabel_;
    QPushButton *applyButton_;
    QPushButton *resetButton_;

    // Parameter widgets (key -> widget)
    std::map<std::string, QWidget*> parameterWidgets_;

    // Re-entrancy guard to prevent concurrent showPluginConfig() calls
    bool isUpdating_;

    // Mutex to protect currentPlugin_ and parameterWidgets_ from concurrent access
    std::mutex stateMutex_;
};

} // namespace nda

#endif // PLUGINSIDEBAR_H

