#ifndef PLUGINSVIEW_H
#define PLUGINSVIEW_H

#include <QWidget>
#include <QListWidget>
#include <QPushButton>
#include <QLabel>

class PluginsView : public QWidget
{
    Q_OBJECT

public:
    explicit PluginsView(QWidget *parent = nullptr);
    ~PluginsView();

private slots:
    void onLoadPlugin();
    void onUnloadPlugin();
    void onPluginSelected(QListWidgetItem *item);
    void onAutoLoadPythonPlugins();

private:
    void setupUI();
    void populatePlugins();

    QListWidget *pluginList;
    QPushButton *loadButton;
    QPushButton *autoLoadButton;
    QPushButton *unloadButton;
    QLabel *pluginInfoLabel;
};

#endif // PLUGINSVIEW_H
