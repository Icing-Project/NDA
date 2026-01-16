#include "ui/PluginSidebar.h"
#include "audio/WasapiDeviceEnum.h"
#include <QCoreApplication>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QMessageBox>
#include <QScrollArea>
#include <algorithm> // For std::transform
#include <functional> // For std::function
#include <iostream>
#include <string>
#include <vector>

namespace nda {

PluginSidebar::PluginSidebar(QWidget *parent)
    : QWidget(parent)
    , isUpdating_(false)  // Initialize re-entrancy guard
{
    setupUI();
    applyModernStyles();
    hide(); // Hidden by default until plugin selected
}

PluginSidebar::~PluginSidebar()
{
}

void PluginSidebar::setupUI()
{
    setObjectName("pluginSidebar");
    setMinimumWidth(300);
    setMaximumWidth(300);

    mainLayout_ = new QVBoxLayout(this);
    mainLayout_->setSpacing(15);
    mainLayout_->setContentsMargins(15, 15, 15, 15);

    // Header
    QLabel *titleLabel = new QLabel("Plugin Configuration", this);
    titleLabel->setObjectName("sidebarTitle");
    titleLabel->setStyleSheet("font-size: 16px; font-weight: 700; color: #ffffff;");
    mainLayout_->addWidget(titleLabel);

    // Plugin name
    pluginNameLabel_ = new QLabel("No plugin selected", this);
    pluginNameLabel_->setObjectName("pluginNameLabel");
    pluginNameLabel_->setWordWrap(true);
    mainLayout_->addWidget(pluginNameLabel_);

    mainLayout_->addSpacing(10);

    // Content area (scrollable)
    QScrollArea *scrollArea = new QScrollArea(this);
    scrollArea->setWidgetResizable(true);
    scrollArea->setFrameShape(QFrame::NoFrame);
    scrollArea->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);

    contentWidget_ = new QWidget();
    QVBoxLayout *contentLayout = new QVBoxLayout(contentWidget_);
    contentLayout->setSpacing(10);
    scrollArea->setWidget(contentWidget_);

    mainLayout_->addWidget(scrollArea, 1);

    // Buttons
    QHBoxLayout *buttonLayout = new QHBoxLayout();

    applyButton_ = new QPushButton("Apply", this);
    applyButton_->setObjectName("paramButton");
    connect(applyButton_, &QPushButton::clicked, this, &PluginSidebar::onApplyClicked);
    buttonLayout->addWidget(applyButton_);

    resetButton_ = new QPushButton("Reset", this);
    resetButton_->setObjectName("paramButton");
    connect(resetButton_, &QPushButton::clicked, this, &PluginSidebar::onResetClicked);
    buttonLayout->addWidget(resetButton_);

    mainLayout_->addLayout(buttonLayout);
}

void PluginSidebar::clearParameters()
{
    // Clear all parameter widgets first (before removing from layout)
    // This prevents accessing stale pointers during layout cleanup
    parameterWidgets_.clear();

    // Clear all widgets from the layout using Qt's safe deletion mechanism
    QLayout *layout = contentWidget_->layout();
    if (!layout) return;

    // Recursively delete all widgets in the layout
    std::function<void(QLayout*)> clearLayoutRecursive = [&](QLayout* l) {
        if (!l) return;

        QLayoutItem *item;
        while ((item = l->takeAt(0)) != nullptr) {
            // Handle nested layouts (e.g., from sliders, file selectors)
            if (QLayout *childLayout = item->layout()) {
                clearLayoutRecursive(childLayout);
                // Don't delete child layout yet - it's owned by the item
            }

            // Handle widgets
            if (QWidget *w = item->widget()) {
                // CRITICAL: Disconnect ALL signals before deletion
                // This prevents lambdas from firing with dangling pointers
                w->disconnect();

                // Remove from parent and schedule for deletion
                w->setParent(nullptr);
                w->deleteLater();
            }

            // Now safe to delete the layout item
            delete item;
        }
    };

    clearLayoutRecursive(layout);
}

void PluginSidebar::showPluginConfig(std::shared_ptr<BasePlugin> plugin)
{
    // Re-entrancy guard - prevent concurrent calls (check before mutex for early exit)
    if (isUpdating_) {
        std::cerr << "[PluginSidebar] Blocked re-entrant call to showPluginConfig" << std::endl;
        return;
    }

    // RAII guard - automatically resets isUpdating_ even if exception thrown
    UpdateGuard guard(isUpdating_);

    // Phase 1: Clear old widgets (mutex-protected)
    {
        std::lock_guard<std::mutex> lock(stateMutex_);

        // Store previous plugin to prevent dangling reference during cleanup
        std::shared_ptr<BasePlugin> previousPlugin = currentPlugin_;
        currentPlugin_ = plugin;

        clearParameters();

        if (!plugin) {
            hide();
            return;
        }
    } // Mutex released here

    // Phase 2: Force widget deletion BEFORE creating new widgets
    // CRITICAL: Must happen OUTSIDE mutex to avoid deadlock from processEvents
    // This ensures WASAPI/COM objects from old AIOC widgets are fully released
    // before creating new AIOC widgets (prevents COM conflicts)
    QCoreApplication::processEvents(QEventLoop::ExcludeUserInputEvents);
    QCoreApplication::processEvents(QEventLoop::ExcludeUserInputEvents);

    // Phase 3: Create new widgets (mutex-protected)
    {
        std::lock_guard<std::mutex> lock(stateMutex_);

        auto info = plugin->getInfo();
        pluginNameLabel_->setText(QString::fromStdString(info.name));

        QVBoxLayout *contentLayout = qobject_cast<QVBoxLayout*>(contentWidget_->layout());
        if (!contentLayout) {
            contentLayout = new QVBoxLayout(contentWidget_);
            contentWidget_->setLayout(contentLayout);
        }

        // Encryptor/Decryptor plugins
        if (info.name.find("Encryptor") != std::string::npos || info.name.find("Decryptor") != std::string::npos) {
            addTextInput("Encryption Key:", "key");
            addFileSelector("Key File:", "key_file");
        }

        if (info.type == PluginType::AudioSink && info.name.find("File") != std::string::npos) {
            addFileSelector("Output File:", "output_path");
        }

        // v2.2: AIOC Sink specific UI
        if (info.name.find("AIOC") != std::string::npos && info.type == PluginType::AudioSink) {
            addAIOCDeviceSelector("AIOC Output Device:", "device_id", 1);  // 1 = eRender (speakers)
            addPTTModeSelector("PTT Mode:", "ptt_mode");
            addSlider("VPTT Threshold:", "vptt_threshold", 0, 32768, 64);
            addSlider("VPTT Hang (ms):", "vptt_hang_ms", 0, 2000, 200);
            addTextInput("CDC Port (auto-detect):", "cdc_port");
        }

        // v2.2: AIOC Source specific UI
        if (info.name.find("AIOC") != std::string::npos && info.type == PluginType::AudioSource) {
            addAIOCDeviceSelector("AIOC Input Device:", "device_id", 0);  // 0 = eCapture (microphones)
            addSlider("VCOS Threshold:", "vcos_threshold", 0, 32768, 32);
            addSlider("VCOS Hang (ms):", "vcos_hang_ms", 0, 2000, 200);
        }

        contentLayout->addStretch();

        show();
    } // Phase 3 mutex released here

    // UpdateGuard destructor automatically resets isUpdating_ = false here
}

void PluginSidebar::addDeviceSelector(const QString& label, const QString& key)
{
    QVBoxLayout *contentLayout = qobject_cast<QVBoxLayout*>(contentWidget_->layout());
    if (!contentLayout) return;

    QLabel *labelWidget = new QLabel(label, this);
    labelWidget->setObjectName("paramLabel");
    contentLayout->addWidget(labelWidget);

    QComboBox *combo = new QComboBox(this);
    combo->setObjectName("paramCombo");
    combo->setMinimumHeight(35);

    // Placeholder devices
    combo->addItem("Default Device");
    combo->addItem("Built-in Microphone");
    combo->addItem("USB Audio Device");

    contentLayout->addWidget(combo);
    parameterWidgets_[key.toStdString()] = combo;
}

void PluginSidebar::addSlider(const QString& label, const QString& key,
                              int min, int max, int defaultVal)
{
    QVBoxLayout *contentLayout = qobject_cast<QVBoxLayout*>(contentWidget_->layout());
    if (!contentLayout) return;

    QLabel *labelWidget = new QLabel(label, this);
    labelWidget->setObjectName("paramLabel");
    contentLayout->addWidget(labelWidget);

    QHBoxLayout *sliderLayout = new QHBoxLayout();

    QSlider *slider = new QSlider(Qt::Horizontal, this);
    slider->setObjectName("paramSlider");
    slider->setRange(min, max);
    slider->setValue(defaultVal);
    sliderLayout->addWidget(slider);

    QLabel *valueLabel = new QLabel(QString::number(defaultVal), this);
    valueLabel->setObjectName("paramValue");
    valueLabel->setMinimumWidth(50);
    sliderLayout->addWidget(valueLabel);

    contentLayout->addLayout(sliderLayout);
    parameterWidgets_[key.toStdString()] = slider;

    // Use QPointer to safely handle widget deletion (prevents heap corruption)
    QPointer<QLabel> safeValueLabel(valueLabel);
    connect(slider, &QSlider::valueChanged, [safeValueLabel](int val) {
        if (safeValueLabel) {  // Automatically becomes null if widget deleted
            safeValueLabel->setText(QString::number(val));
        }
    });
}

void PluginSidebar::addFileSelector(const QString& label, const QString& key)
{
    QVBoxLayout *contentLayout = qobject_cast<QVBoxLayout*>(contentWidget_->layout());
    if (!contentLayout) return;

    QLabel *labelWidget = new QLabel(label, this);
    labelWidget->setObjectName("paramLabel");
    contentLayout->addWidget(labelWidget);

    QHBoxLayout *fileLayout = new QHBoxLayout();

    QLineEdit *pathEdit = new QLineEdit(this);
    pathEdit->setObjectName("paramLineEdit");
    pathEdit->setPlaceholderText("Select file...");
    fileLayout->addWidget(pathEdit);

    QPushButton *browseBtn = new QPushButton("Browse", this);
    browseBtn->setObjectName("paramButton");
    browseBtn->setMaximumWidth(80);
    fileLayout->addWidget(browseBtn);

    contentLayout->addLayout(fileLayout);
    parameterWidgets_[key.toStdString()] = pathEdit;

    // Use QPointer to safely handle widget deletion (prevents heap corruption)
    QPointer<QLineEdit> safePathEdit(pathEdit);
    connect(browseBtn, &QPushButton::clicked, [this, safePathEdit]() {
        if (!safePathEdit) return;  // Widget was deleted
        QString file = QFileDialog::getSaveFileName(this, "Select File", "", "WAV Files (*.wav)");
        if (!file.isEmpty() && safePathEdit) {  // Check again after dialog closes
            safePathEdit->setText(file);
        }
    });
}

void PluginSidebar::addCheckbox(const QString& label, const QString& key)
{
    QVBoxLayout *contentLayout = qobject_cast<QVBoxLayout*>(contentWidget_->layout());
    if (!contentLayout) return;

    QCheckBox *checkbox = new QCheckBox(label, this);
    checkbox->setObjectName("paramCheckbox");
    contentLayout->addWidget(checkbox);
    parameterWidgets_[key.toStdString()] = checkbox;
}

void PluginSidebar::addTextInput(const QString& label, const QString& key)
{
    QVBoxLayout *contentLayout = qobject_cast<QVBoxLayout*>(contentWidget_->layout());
    if (!contentLayout) return;

    QLabel *labelWidget = new QLabel(label, this);
    labelWidget->setObjectName("paramLabel");
    contentLayout->addWidget(labelWidget);

    QLineEdit *lineEdit = new QLineEdit(this);
    lineEdit->setObjectName("paramLineEdit");
    contentLayout->addWidget(lineEdit);
    parameterWidgets_[key.toStdString()] = lineEdit;
}

// v2.2: AIOC device selector with WASAPI enumeration
void PluginSidebar::addAIOCDeviceSelector(const QString& label, const QString& key, int direction)
{
    QVBoxLayout *contentLayout = qobject_cast<QVBoxLayout*>(contentWidget_->layout());
    if (!contentLayout) return;

    QLabel *labelWidget = new QLabel(label, this);
    labelWidget->setObjectName("paramLabel");
    contentLayout->addWidget(labelWidget);

    QComboBox *combo = new QComboBox(this);
    combo->setObjectName("paramCombo");
    combo->setMinimumHeight(35);

    // Enumerate WASAPI devices
    // direction: 0 = eCapture (microphones), 1 = eRender (speakers)
    auto devices = enumerateWASAPIDevices(direction);

    int aiocIndex = -1;
    for (size_t i = 0; i < devices.size(); ++i) {
        combo->addItem(QString::fromStdString(devices[i].friendlyName),
                       QString::fromStdString(devices[i].id));

        // Pre-select first device containing "aioc" (case-insensitive)
        if (aiocIndex == -1) {
            std::string name = devices[i].friendlyName;
            std::transform(name.begin(), name.end(), name.begin(), ::tolower);
            if (name.find("aioc") != std::string::npos) {
                aiocIndex = static_cast<int>(i);
            }
        }
    }

    if (aiocIndex >= 0) {
        combo->setCurrentIndex(aiocIndex);
    }

    contentLayout->addWidget(combo);
    parameterWidgets_[key.toStdString()] = combo;
}

// v2.2: PTT mode selector for AIOC
void PluginSidebar::addPTTModeSelector(const QString& label, const QString& key)
{
    QVBoxLayout *contentLayout = qobject_cast<QVBoxLayout*>(contentWidget_->layout());
    if (!contentLayout) return;

    QLabel *labelWidget = new QLabel(label, this);
    labelWidget->setObjectName("paramLabel");
    contentLayout->addWidget(labelWidget);

    QComboBox *combo = new QComboBox(this);
    combo->setObjectName("paramCombo");
    combo->setMinimumHeight(35);

    // PTT mode options with userData for parameter value
    combo->addItem("HID Manual (Recommended)", "hid_manual");
    combo->addItem("CDC Manual (COM Port)", "cdc_manual");
    combo->addItem("VPTT Auto (Voice Activated)", "vptt_auto");

    // Default to HID Manual (index 0)
    combo->setCurrentIndex(0);

    contentLayout->addWidget(combo);
    parameterWidgets_[key.toStdString()] = combo;
}

void PluginSidebar::onApplyClicked()
{
    // Emit signal to cancel any pending plugin selection changes
    emit aboutToApplyParameters();

    // Thread-safe mutex lock
    std::lock_guard<std::mutex> lock(stateMutex_);

    // Store local copy to prevent TOCTOU (time-of-check-time-of-use) race condition
    std::shared_ptr<BasePlugin> plugin = currentPlugin_;

    if (!plugin) {
        QMessageBox::warning(this, "Configuration Error",
            "No plugin selected. Please select a plugin first.");
        return;
    }

    // Validate plugin is still in valid state
    try {
        auto info = plugin->getInfo();  // Test if plugin is still valid
    } catch (const std::exception& e) {
        QMessageBox::critical(this, "Configuration Error",
            QString("Plugin is no longer valid: %1\nPlease reselect the plugin.")
                .arg(e.what()));
        currentPlugin_.reset();
        clearParameters();
        hide();
        return;
    } catch (...) {
        QMessageBox::critical(this, "Configuration Error",
            "Plugin is no longer valid. Please reselect the plugin.");
        currentPlugin_.reset();
        clearParameters();
        hide();
        return;
    }

    // Apply parameters with error handling
    for (const auto& [key, widget] : parameterWidgets_) {
        QString value;

        if (auto* combo = qobject_cast<QComboBox*>(widget)) {
            // v2.2: Prefer userData (device GUID, PTT mode string) over display text
            value = combo->currentData().toString();
            if (value.isEmpty()) {
                value = combo->currentText();
            }
        } else if (auto* slider = qobject_cast<QSlider*>(widget)) {
            value = QString::number(slider->value());
        } else if (auto* lineEdit = qobject_cast<QLineEdit*>(widget)) {
            value = lineEdit->text();
        } else if (auto* checkbox = qobject_cast<QCheckBox*>(widget)) {
            value = checkbox->isChecked() ? "true" : "false";
        }

        try {
            plugin->setParameter(key, value.toStdString());
            emit parameterChanged(key, value.toStdString());
        } catch (const std::exception& e) {
            QMessageBox::critical(this, "Configuration Error",
                QString("Failed to set parameter '%1': %2")
                    .arg(QString::fromStdString(key))
                    .arg(e.what()));
            return;
        } catch (...) {
            QMessageBox::critical(this, "Configuration Error",
                QString("Failed to set parameter '%1'. Plugin may be in invalid state.")
                    .arg(QString::fromStdString(key)));
            return;
        }
    }

    // Success feedback
    QMessageBox::information(this, "Success",
        "Plugin parameters applied successfully.");
}

void PluginSidebar::onResetClicked()
{
    // Reset all widgets to default values
    for (const auto& [key, widget] : parameterWidgets_) {
        if (auto* slider = qobject_cast<QSlider*>(widget)) {
            slider->setValue(0);
        } else if (auto* lineEdit = qobject_cast<QLineEdit*>(widget)) {
            lineEdit->clear();
        } else if (auto* checkbox = qobject_cast<QCheckBox*>(widget)) {
            checkbox->setChecked(false);
        }
    }
}

void PluginSidebar::applyModernStyles()
{
    setStyleSheet(R"(
        #pluginSidebar {
            background-color: #16213e;
            border-left: 1px solid rgba(255, 255, 255, 0.08);
        }

        #sidebarTitle {
            font-size: 16px;
            font-weight: 700;
            color: #ffffff;
        }

        #pluginNameLabel {
            font-size: 14px;
            color: #94a3b8;
        }

        #paramLabel {
            font-size: 12px;
            font-weight: 600;
            color: #cbd5e1;
            margin-top: 5px;
        }

        #paramCombo {
            background-color: #2a2a3e;
            border: 1px solid #3a3a4e;
            border-radius: 6px;
            padding: 8px;
            color: #eaeaea;
            font-size: 13px;
        }

        #paramLineEdit {
            background-color: #2a2a3e;
            border: 1px solid #3a3a4e;
            border-radius: 6px;
            padding: 8px;
            color: #eaeaea;
            font-size: 13px;
        }

        #paramButton {
            background-color: #475569;
            color: white;
            border: none;
            border-radius: 6px;
            font-size: 13px;
            font-weight: 600;
            padding: 8px 15px;
            min-height: 35px;
        }

        #paramButton:hover {
            background-color: #64748b;
        }

        #paramSlider::groove:horizontal {
            background: #2a2a3e;
            height: 6px;
            border-radius: 3px;
        }

        #paramSlider::handle:horizontal {
            background: #3b82f6;
            width: 16px;
            height: 16px;
            border-radius: 8px;
            margin: -5px 0;
        }

        #paramValue {
            font-size: 13px;
            color: #cbd5e1;
        }

        #paramCheckbox {
            color: #cbd5e1;
            font-size: 13px;
        }
    )");
}

} // namespace nda

