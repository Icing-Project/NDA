#ifndef ABOUTDIALOG_H
#define ABOUTDIALOG_H

#include "core/ProcessingPipeline.h"
#include "plugins/PluginTypes.h"
#include <QDialog>
#include <memory>

namespace nda {
class PluginManager;
}

class QPlainTextEdit;
class QWidget;

class AboutDialog : public QDialog
{
    Q_OBJECT

public:
    explicit AboutDialog(std::shared_ptr<nda::PluginManager> pluginManager,
                         std::shared_ptr<nda::ProcessingPipeline> txPipeline,
                         std::shared_ptr<nda::ProcessingPipeline> rxPipeline,
                         QWidget* parent = nullptr);
    ~AboutDialog() override = default;

private slots:
    void onOpenStartHere();
    void onOpenInstallation();
    void onOpenTroubleshooting();
    void onOpenPluginDiscovery();
    void onCopyDiagnostics();

private:
    void setupUI();
    QWidget* createOverviewTab();
    QWidget* createRuntimeTab();
    QWidget* createPluginsTab();
    QWidget* createDocsTab();

    QString buildRuntimeReport() const;
    QString buildPluginReport() const;
    QString buildDiagnosticsSnapshot() const;

    QString resolveDocPath(const QString& relativePath) const;
    bool openDoc(const QString& relativePath, const QString& title);

    static QString pipelineHealthToString(nda::ProcessingPipeline::HealthStatus status);
    static QString pluginTypeToString(nda::PluginType type);

    std::shared_ptr<nda::PluginManager> pluginManager_;
    std::shared_ptr<nda::ProcessingPipeline> txPipeline_;
    std::shared_ptr<nda::ProcessingPipeline> rxPipeline_;

    QPlainTextEdit* runtimeText_;
    QPlainTextEdit* pluginsText_;
};

#endif // ABOUTDIALOG_H
