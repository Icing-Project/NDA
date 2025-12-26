#include "ui/PluginSidebar.h"
#include <QScrollArea>
#include <QGroupBox>
#include <QHBoxLayout>

namespace nda {

PluginSidebar::PluginSidebar(QWidget *parent)
    : QWidget(parent)
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
    // Clear all parameter widgets
    QLayout *layout = contentWidget_->layout();
    if (layout) {
        QLayoutItem *item;
        while ((item = layout->takeAt(0)) != nullptr) {
            delete item->widget();
            delete item;
        }
    }
    parameterWidgets_.clear();
}

void PluginSidebar::showPluginConfig(std::shared_ptr<BasePlugin> plugin)
{
    currentPlugin_ = plugin;
    clearParameters();
    
    if (!plugin) {
        hide();
        return;
    }
    
    auto info = plugin->getInfo();
    pluginNameLabel_->setText(QString::fromStdString(info.name));
    
    QVBoxLayout *contentLayout = qobject_cast<QVBoxLayout*>(contentWidget_->layout());
    if (!contentLayout) {
        contentLayout = new QVBoxLayout(contentWidget_);
        contentWidget_->setLayout(contentLayout);
    }
    
    // Example parameters based on plugin type
    if (info.type == PluginType::AudioSource) {
        addDeviceSelector("Audio Device:", "device_name");
        addSlider("Gain (dB):", "gain_db", -20, 20, 0);
        addCheckbox("Enable AGC:", "enable_agc");
    }
    
    if (info.name.find("Encryptor") != std::string::npos || info.name.find("Decryptor") != std::string::npos) {
        addTextInput("Encryption Key:", "key");
        addFileSelector("Key File:", "key_file");
    }
    
    if (info.type == PluginType::AudioSink && info.name.find("File") != std::string::npos) {
        addFileSelector("Output File:", "output_path");
    }
    
    contentLayout->addStretch();
    
    show();
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
    
    connect(slider, &QSlider::valueChanged, [valueLabel](int val) {
        valueLabel->setText(QString::number(val));
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
    
    connect(browseBtn, &QPushButton::clicked, [this, pathEdit]() {
        QString file = QFileDialog::getSaveFileName(this, "Select File", "", "WAV Files (*.wav)");
        if (!file.isEmpty()) {
            pathEdit->setText(file);
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

void PluginSidebar::onApplyClicked()
{
    if (!currentPlugin_) return;
    
    for (const auto& [key, widget] : parameterWidgets_) {
        QString value;
        
        if (auto* combo = qobject_cast<QComboBox*>(widget)) {
            value = combo->currentText();
        } else if (auto* slider = qobject_cast<QSlider*>(widget)) {
            value = QString::number(slider->value());
        } else if (auto* lineEdit = qobject_cast<QLineEdit*>(widget)) {
            value = lineEdit->text();
        } else if (auto* checkbox = qobject_cast<QCheckBox*>(widget)) {
            value = checkbox->isChecked() ? "true" : "false";
        }
        
        currentPlugin_->setParameter(key, value.toStdString());
        emit parameterChanged(key, value.toStdString());
    }
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

