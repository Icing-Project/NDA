#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QTabWidget>
#include <QMenuBar>
#include <QStatusBar>
#include "core/ProcessingPipeline.h"
#include "plugins/PluginManager.h"
#include <memory>

class Dashboard;
class PipelineView;
class SettingsView;

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void onPipelineStarted();
    void onPipelineStopped();
    void onStatusUpdate(const QString &message);

private:
    void setupUI();
    void createMenus();
    void createStatusBar();

    QTabWidget *tabWidget;
    Dashboard *dashboard;
    PipelineView *pipelineView;
    SettingsView *settingsView;

    // Core components
    std::shared_ptr<nda::PluginManager> pluginManager_;
    std::shared_ptr<nda::ProcessingPipeline> pipeline_;
};

#endif // MAINWINDOW_H
