#include "ui/AboutDialog.h"
#include "plugins/BasePlugin.h"
#include "plugins/PluginManager.h"
#include "plugins/PluginPaths.h"
#include <QApplication>
#include <QClipboard>
#include <QCoreApplication>
#include <QDateTime>
#include <QDesktopServices>
#include <QDialogButtonBox>
#include <QDir>
#include <QFileInfo>
#include <QHBoxLayout>
#include <QLabel>
#include <QMessageBox>
#include <QPlainTextEdit>
#include <QPushButton>
#include <QStringList>
#include <QSysInfo>
#include <QTabWidget>
#include <QUrl>
#include <QVBoxLayout>
#include <utility>

namespace {

QString boolToWord(bool value)
{
    return value ? "yes" : "no";
}

} // namespace

AboutDialog::AboutDialog(std::shared_ptr<nda::PluginManager> pluginManager,
                         std::shared_ptr<nda::ProcessingPipeline> txPipeline,
                         std::shared_ptr<nda::ProcessingPipeline> rxPipeline,
                         QWidget* parent)
    : QDialog(parent)
    , pluginManager_(std::move(pluginManager))
    , txPipeline_(std::move(txPipeline))
    , rxPipeline_(std::move(rxPipeline))
    , runtimeText_(nullptr)
    , pluginsText_(nullptr)
{
    setupUI();
}

void AboutDialog::setupUI()
{
    setWindowTitle("About NDA");
    setMinimumSize(860, 620);

    QVBoxLayout* mainLayout = new QVBoxLayout(this);

    QTabWidget* tabs = new QTabWidget(this);
    tabs->addTab(createOverviewTab(), "Overview");
    tabs->addTab(createRuntimeTab(), "Runtime");
    tabs->addTab(createPluginsTab(), "Plugins");
    tabs->addTab(createDocsTab(), "Docs & Support");
    mainLayout->addWidget(tabs);

    QDialogButtonBox* buttons = new QDialogButtonBox(QDialogButtonBox::Close, this);
    connect(buttons, &QDialogButtonBox::rejected, this, &QDialog::reject);
    mainLayout->addWidget(buttons);
}

QWidget* AboutDialog::createOverviewTab()
{
    QWidget* tab = new QWidget(this);
    QVBoxLayout* layout = new QVBoxLayout(tab);

    QLabel* summaryLabel = new QLabel(
        "<b>NDA</b> is a real-time audio encryption bridge with dual independent pipelines.",
        tab);
    summaryLabel->setWordWrap(true);
    layout->addWidget(summaryLabel);

    QLabel* pipelineLabel = new QLabel(
        "<pre>"
        "TX: Source -> [Processor optional] -> Sink\n"
        "RX: Source -> [Processor optional] -> Sink"
        "</pre>",
        tab);
    pipelineLabel->setTextFormat(Qt::RichText);
    pipelineLabel->setWordWrap(true);
    layout->addWidget(pipelineLabel);

    QLabel* modelLabel = new QLabel(
        "Core model: 3-slot audio pipeline, with encryption handled by processor plugins.",
        tab);
    modelLabel->setWordWrap(true);
    layout->addWidget(modelLabel);

    QLabel* scopeLabel = new QLabel(
        "Scope boundaries:\n"
        "- Network transport is external to NDA core (AIOC, VB-Cable, VoIP apps).\n"
        "- Key exchange is out-of-band; NDA manages key material and plugin parameters.",
        tab);
    scopeLabel->setWordWrap(true);
    layout->addWidget(scopeLabel);

    QLabel* usageLabel = new QLabel(
        "Useful controls:\n"
        "- Radio Mode: one-click AIOC routing preset.\n"
        "- Start Both / Stop Both: coordinated TX/RX lifecycle.\n"
        "- PTT hotkeys (AIOC sink): hold T or Space.",
        tab);
    usageLabel->setWordWrap(true);
    layout->addWidget(usageLabel);

    layout->addStretch();
    return tab;
}

QWidget* AboutDialog::createRuntimeTab()
{
    QWidget* tab = new QWidget(this);
    QVBoxLayout* layout = new QVBoxLayout(tab);

    QLabel* description = new QLabel(
        "Live runtime/build snapshot for support and troubleshooting.",
        tab);
    description->setWordWrap(true);
    layout->addWidget(description);

    runtimeText_ = new QPlainTextEdit(tab);
    runtimeText_->setReadOnly(true);
    runtimeText_->setLineWrapMode(QPlainTextEdit::NoWrap);
    runtimeText_->setPlainText(buildRuntimeReport());
    layout->addWidget(runtimeText_, 1);

    QHBoxLayout* buttonLayout = new QHBoxLayout();
    QPushButton* refreshButton = new QPushButton("Refresh", tab);
    connect(refreshButton, &QPushButton::clicked, this, [this]() {
        runtimeText_->setPlainText(buildRuntimeReport());
    });
    buttonLayout->addWidget(refreshButton);

    QPushButton* copyButton = new QPushButton("Copy Diagnostics", tab);
    connect(copyButton, &QPushButton::clicked, this, &AboutDialog::onCopyDiagnostics);
    buttonLayout->addWidget(copyButton);

    buttonLayout->addStretch();
    layout->addLayout(buttonLayout);

    return tab;
}

QWidget* AboutDialog::createPluginsTab()
{
    QWidget* tab = new QWidget(this);
    QVBoxLayout* layout = new QVBoxLayout(tab);

    QLabel* description = new QLabel(
        "Loaded plugins and discovery path health (including NDA_PLUGIN_PATH override).",
        tab);
    description->setWordWrap(true);
    layout->addWidget(description);

    pluginsText_ = new QPlainTextEdit(tab);
    pluginsText_->setReadOnly(true);
    pluginsText_->setLineWrapMode(QPlainTextEdit::NoWrap);
    pluginsText_->setPlainText(buildPluginReport());
    layout->addWidget(pluginsText_, 1);

    QHBoxLayout* buttonLayout = new QHBoxLayout();
    QPushButton* refreshButton = new QPushButton("Refresh", tab);
    connect(refreshButton, &QPushButton::clicked, this, [this]() {
        pluginsText_->setPlainText(buildPluginReport());
    });
    buttonLayout->addWidget(refreshButton);

    QPushButton* copyButton = new QPushButton("Copy Diagnostics", tab);
    connect(copyButton, &QPushButton::clicked, this, &AboutDialog::onCopyDiagnostics);
    buttonLayout->addWidget(copyButton);

    buttonLayout->addStretch();
    layout->addLayout(buttonLayout);

    return tab;
}

QWidget* AboutDialog::createDocsTab()
{
    QWidget* tab = new QWidget(this);
    QVBoxLayout* layout = new QVBoxLayout(tab);

    QLabel* description = new QLabel(
        "Open local documentation quickly and copy a support snapshot.",
        tab);
    description->setWordWrap(true);
    layout->addWidget(description);

    QPushButton* startHereButton = new QPushButton("Open START_HERE", tab);
    connect(startHereButton, &QPushButton::clicked, this, &AboutDialog::onOpenStartHere);
    layout->addWidget(startHereButton);

    QPushButton* installationButton = new QPushButton("Open Installation Guide", tab);
    connect(installationButton, &QPushButton::clicked, this, &AboutDialog::onOpenInstallation);
    layout->addWidget(installationButton);

    QPushButton* troubleshootingButton = new QPushButton("Open Troubleshooting Guide", tab);
    connect(troubleshootingButton, &QPushButton::clicked, this, &AboutDialog::onOpenTroubleshooting);
    layout->addWidget(troubleshootingButton);

    QPushButton* pluginDiscoveryButton = new QPushButton("Open Plugin Discovery Guide", tab);
    connect(pluginDiscoveryButton, &QPushButton::clicked, this, &AboutDialog::onOpenPluginDiscovery);
    layout->addWidget(pluginDiscoveryButton);

    QPushButton* copyButton = new QPushButton("Copy Diagnostics", tab);
    connect(copyButton, &QPushButton::clicked, this, &AboutDialog::onCopyDiagnostics);
    layout->addWidget(copyButton);

    QLabel* noteLabel = new QLabel(
        "If a document cannot be found, NDA will show all searched locations.",
        tab);
    noteLabel->setWordWrap(true);
    layout->addWidget(noteLabel);

    layout->addStretch();
    return tab;
}

QString AboutDialog::buildRuntimeReport() const
{
    QString version = QCoreApplication::applicationVersion();
    if (version.isEmpty()) {
        version = "unknown";
    }

    QStringList lines;
    lines << "Application";
    lines << "-----------";
    lines << QString("Name: %1").arg(QCoreApplication::applicationName());
    lines << QString("Version: %1").arg(version);
    lines << QString("Organization: %1").arg(QCoreApplication::organizationName());
    lines << QString("Qt runtime version: %1").arg(qVersion());
    lines << QString("Build timestamp: %1 %2").arg(__DATE__).arg(__TIME__);
    lines << QString("OS: %1").arg(QSysInfo::prettyProductName());
    lines << QString("Kernel: %1 %2")
                 .arg(QSysInfo::kernelType())
                 .arg(QSysInfo::kernelVersion());
    lines << QString("CPU architecture: %1").arg(QSysInfo::currentCpuArchitecture());

#ifdef NDA_ENABLE_PYTHON
    lines << "Python plugin support: ENABLED";
#else
    lines << "Python plugin support: DISABLED";
#endif
    lines << QString("Plugin API version: %1").arg(NDA_PLUGIN_API_VERSION);

    lines << "";
    lines << "Pipeline State";
    lines << "--------------";

    auto appendPipeline = [this, &lines](const QString& name,
                                          const std::shared_ptr<nda::ProcessingPipeline>& pipeline) {
        if (!pipeline) {
            lines << QString("%1 available: no").arg(name);
            return;
        }

        const bool running = pipeline->isRunning();
        lines << QString("%1 available: yes").arg(name);
        lines << QString("%1 running: %2").arg(name).arg(boolToWord(running));

        if (running) {
            lines << QString("%1 health: %2")
                         .arg(name)
                         .arg(pipelineHealthToString(pipeline->getHealthStatus()));
            lines << QString("%1 latency: %2 ms")
                         .arg(name)
                         .arg(QString::number(pipeline->getLatency(), 'f', 1));
            lines << QString("%1 CPU: %2%%")
                         .arg(name)
                         .arg(QString::number(static_cast<int>(pipeline->getCPULoad())));
            lines << QString("%1 drift: %2 ms (max %3 ms)")
                         .arg(name)
                         .arg(QString::number(pipeline->getCurrentDriftMs(), 'f', 1))
                         .arg(QString::number(pipeline->getMaxDriftMs(), 'f', 1));
            lines << QString("%1 failures: read=%2 write=%3")
                         .arg(name)
                         .arg(pipeline->getReadFailures())
                         .arg(pipeline->getWriteFailures());
        }
    };

    appendPipeline("TX", txPipeline_);
    appendPipeline("RX", rxPipeline_);

    return lines.join('\n');
}

QString AboutDialog::buildPluginReport() const
{
    QStringList lines;
    lines << "Plugin Inventory";
    lines << "----------------";

    if (!pluginManager_) {
        lines << "Plugin manager available: no";
    } else {
        const auto allPlugins = pluginManager_->getAllPlugins();
        const auto sourcePlugins = pluginManager_->getPluginsByType(nda::PluginType::AudioSource);
        const auto processorPlugins = pluginManager_->getPluginsByType(nda::PluginType::Processor);
        const auto sinkPlugins = pluginManager_->getPluginsByType(nda::PluginType::AudioSink);

        int cppCount = 0;
        int pythonCount = 0;
        for (const auto& plugin : allPlugins) {
            const QString path = QString::fromStdString(plugin.path);
            if (path.endsWith(".py", Qt::CaseInsensitive)) {
                pythonCount++;
            } else {
                cppCount++;
            }
        }

        lines << "Plugin manager available: yes";
        lines << QString("Total loaded: %1").arg(allPlugins.size());
        lines << QString("C++ loaded: %1").arg(cppCount);
        lines << QString("Python loaded: %1").arg(pythonCount);
        lines << QString("Sources: %1").arg(sourcePlugins.size());
        lines << QString("Processors: %1").arg(processorPlugins.size());
        lines << QString("Sinks: %1").arg(sinkPlugins.size());

        lines << "";
        lines << "Loaded Plugin Details";
        lines << "---------------------";

        if (allPlugins.empty()) {
            lines << "(none)";
        } else {
            for (const auto& plugin : allPlugins) {
                const QString path = QString::fromStdString(plugin.path);
                const QString backend = path.endsWith(".py", Qt::CaseInsensitive) ? "PY" : "CPP";
                lines << QString("[%1] %2 | type=%3 | version=%4 | author=%5")
                             .arg(backend)
                             .arg(QString::fromStdString(plugin.info.name))
                             .arg(pluginTypeToString(plugin.info.type))
                             .arg(QString::fromStdString(plugin.info.version))
                             .arg(QString::fromStdString(plugin.info.author));
                lines << QString("    path: %1").arg(QDir::toNativeSeparators(path));
            }
        }
    }

    const QString envPluginPath = qEnvironmentVariable("NDA_PLUGIN_PATH");
    lines << "";
    lines << "Discovery Configuration";
    lines << "-----------------------";
    lines << QString("NDA_PLUGIN_PATH: %1")
                 .arg(envPluginPath.isEmpty() ? "<unset>" : QDir::toNativeSeparators(envPluginPath));

    lines << "";
    lines << "C++ Search Paths";
    lines << "----------------";
    for (const QString& path : nda::PluginPaths::getCppPluginSearchPaths()) {
        lines << QString("[%1] %2")
                     .arg(QDir(path).exists() ? "OK" : "MISSING")
                     .arg(QDir::toNativeSeparators(path));
    }

    lines << "";
    lines << "Python Search Paths";
    lines << "-------------------";
    for (const QString& path : nda::PluginPaths::getPythonPluginSearchPaths()) {
        lines << QString("[%1] %2")
                     .arg(QDir(path).exists() ? "OK" : "MISSING")
                     .arg(QDir::toNativeSeparators(path));
    }

    return lines.join('\n');
}

QString AboutDialog::buildDiagnosticsSnapshot() const
{
    QStringList lines;
    lines << "NDA Diagnostics Snapshot";
    lines << "========================";
    lines << QString("Captured at: %1")
                 .arg(QDateTime::currentDateTime().toString(Qt::ISODate));
    lines << "";
    lines << buildRuntimeReport();
    lines << "";
    lines << buildPluginReport();
    lines << "";
    lines << "Environment";
    lines << "-----------";

    const QStringList envVars = {
        "NDA_PLUGIN_PATH",
        "NDA_PROFILE",
        "NDA_PROFILE_PIPELINE",
        "NDA_PROFILE_PYBRIDGE",
        "NDA_RESAMPLER_QUALITY",
        "NDA_PIPELINE_FRAME_SIZE",
        "NDA_PIPELINE_BACKPRESSURE_MODE"
    };

    for (const QString& name : envVars) {
        const QByteArray utf8 = name.toUtf8();
        const bool isSet = qEnvironmentVariableIsSet(utf8.constData());
        lines << QString("%1=%2")
                     .arg(name)
                     .arg(isSet ? qEnvironmentVariable(utf8.constData()) : "<unset>");
    }

    return lines.join('\n');
}

QString AboutDialog::resolveDocPath(const QString& relativePath) const
{
    const QString appDir = QCoreApplication::applicationDirPath();
    const QStringList roots = {
        appDir + "/docs",
        appDir + "/../docs",
        appDir + "/../../docs",
        QDir::currentPath() + "/docs"
    };

    const QString fallbackName = QFileInfo(relativePath).fileName();

    for (const QString& root : roots) {
        const QString candidate = QDir(root).absoluteFilePath(relativePath);
        if (QFileInfo::exists(candidate)) {
            return candidate;
        }

        if (fallbackName != relativePath) {
            const QString fallback = QDir(root).absoluteFilePath(fallbackName);
            if (QFileInfo::exists(fallback)) {
                return fallback;
            }
        }
    }

    return QString();
}

bool AboutDialog::openDoc(const QString& relativePath, const QString& title)
{
    const QString resolvedPath = resolveDocPath(relativePath);
    if (resolvedPath.isEmpty()) {
        const QString appDir = QCoreApplication::applicationDirPath();
        const QStringList roots = {
            appDir + "/docs",
            appDir + "/../docs",
            appDir + "/../../docs",
            QDir::currentPath() + "/docs"
        };

        QStringList searched;
        const QString fallbackName = QFileInfo(relativePath).fileName();
        for (const QString& root : roots) {
            searched << QDir::toNativeSeparators(QDir(root).absoluteFilePath(relativePath));
            if (fallbackName != relativePath) {
                searched << QDir::toNativeSeparators(QDir(root).absoluteFilePath(fallbackName));
            }
        }

        QMessageBox::warning(
            this,
            title,
            QStringLiteral("Could not locate the requested document.\n\nSearched:\n") +
                searched.join("\n"));
        return false;
    }

    if (!QDesktopServices::openUrl(QUrl::fromLocalFile(resolvedPath))) {
        QMessageBox::warning(
            this,
            title,
            QStringLiteral("Failed to open document:\n") +
                QDir::toNativeSeparators(resolvedPath));
        return false;
    }

    return true;
}

QString AboutDialog::pipelineHealthToString(nda::ProcessingPipeline::HealthStatus status)
{
    switch (status) {
    case nda::ProcessingPipeline::HealthStatus::OK:
        return "OK";
    case nda::ProcessingPipeline::HealthStatus::Degraded:
        return "Degraded";
    case nda::ProcessingPipeline::HealthStatus::Failing:
        return "Failing";
    }
    return "Unknown";
}

QString AboutDialog::pluginTypeToString(nda::PluginType type)
{
    switch (type) {
    case nda::PluginType::AudioSource:
        return "source";
    case nda::PluginType::Processor:
        return "processor";
    case nda::PluginType::AudioSink:
        return "sink";
    }
    return "unknown";
}

void AboutDialog::onOpenStartHere()
{
    openDoc("START_HERE.md", "Open START_HERE");
}

void AboutDialog::onOpenInstallation()
{
    openDoc("guides/installation.md", "Open Installation Guide");
}

void AboutDialog::onOpenTroubleshooting()
{
    openDoc("guides/troubleshooting.md", "Open Troubleshooting Guide");
}

void AboutDialog::onOpenPluginDiscovery()
{
    openDoc("technical/plugin-discovery.md", "Open Plugin Discovery Guide");
}

void AboutDialog::onCopyDiagnostics()
{
    if (runtimeText_) {
        runtimeText_->setPlainText(buildRuntimeReport());
    }
    if (pluginsText_) {
        pluginsText_->setPlainText(buildPluginReport());
    }

    QApplication::clipboard()->setText(buildDiagnosticsSnapshot());
    QMessageBox::information(this, "Diagnostics Copied",
                             "Copied runtime diagnostics to the clipboard.");
}
