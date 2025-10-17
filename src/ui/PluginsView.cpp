#include "ui/PluginsView.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGroupBox>
#include <QFileDialog>
#include <QMessageBox>
#include <QDir>
#include <QDirIterator>

PluginsView::PluginsView(QWidget *parent)
    : QWidget(parent)
{
    setupUI();
    populatePlugins();
}

PluginsView::~PluginsView()
{
}

void PluginsView::setupUI()
{
    QVBoxLayout *mainLayout = new QVBoxLayout(this);

    // Plugin list
    QGroupBox *listGroup = new QGroupBox("Installed Plugins", this);
    QVBoxLayout *listLayout = new QVBoxLayout(listGroup);

    pluginList = new QListWidget(this);
    connect(pluginList, &QListWidget::itemClicked, this, &PluginsView::onPluginSelected);
    listLayout->addWidget(pluginList);

    QHBoxLayout *buttonLayout = new QHBoxLayout();

    loadButton = new QPushButton("Load Plugin...", this);
    connect(loadButton, &QPushButton::clicked, this, &PluginsView::onLoadPlugin);
    buttonLayout->addWidget(loadButton);

    autoLoadButton = new QPushButton("Auto-Load Python Plugins", this);
    connect(autoLoadButton, &QPushButton::clicked, this, &PluginsView::onAutoLoadPythonPlugins);
    buttonLayout->addWidget(autoLoadButton);

    unloadButton = new QPushButton("Unload Selected", this);
    unloadButton->setEnabled(false);
    connect(unloadButton, &QPushButton::clicked, this, &PluginsView::onUnloadPlugin);
    buttonLayout->addWidget(unloadButton);

    listLayout->addLayout(buttonLayout);

    mainLayout->addWidget(listGroup);

    // Plugin info
    QGroupBox *infoGroup = new QGroupBox("Plugin Information", this);
    QVBoxLayout *infoLayout = new QVBoxLayout(infoGroup);

    pluginInfoLabel = new QLabel("No plugin selected", this);
    pluginInfoLabel->setWordWrap(true);
    infoLayout->addWidget(pluginInfoLabel);

    mainLayout->addWidget(infoGroup);

    mainLayout->addStretch();
}

void PluginsView::populatePlugins()
{
    pluginList->clear();

    // Placeholder - will be populated with real plugins
    pluginList->addItem("Spotify Source (Active)");
    pluginList->addItem("YouTube Music Source (Inactive)");
    pluginList->addItem("SoundCloud Source (Inactive)");
    pluginList->addItem("TIDAL Source (Inactive)");
}

void PluginsView::onLoadPlugin()
{
    QString fileName = QFileDialog::getOpenFileName(this,
        "Load Plugin", "", "Plugin Files (*.dll *.so);;All Files (*)");

    if (!fileName.isEmpty()) {
        pluginList->addItem(fileName);
        pluginInfoLabel->setText("Loaded: " + fileName);
    }
}

void PluginsView::onUnloadPlugin()
{
    QListWidgetItem *item = pluginList->currentItem();
    if (item) {
        delete item;
        pluginInfoLabel->setText("No plugin selected");
        unloadButton->setEnabled(false);
    }
}

void PluginsView::onPluginSelected(QListWidgetItem *item)
{
    if (item) {
        unloadButton->setEnabled(true);
        pluginInfoLabel->setText(
            "Plugin: " + item->text() + "\n\n"
            "Type: Audio Source\n"
            "Version: 1.0.0\n"
            "Status: Active\n"
            "API: NADE Plugin v1.0"
        );
    }
}

void PluginsView::onAutoLoadPythonPlugins()
{
    // Define the plugins_py directory path
    QString pluginsDir = QDir::currentPath() + "/plugins_py";

    QDir dir(pluginsDir);
    if (!dir.exists()) {
        // Try parent directory
        pluginsDir = QDir::currentPath() + "/../plugins_py";
        dir.setPath(pluginsDir);

        if (!dir.exists()) {
            QMessageBox::warning(this, "Directory Not Found",
                "Could not find plugins_py directory at:\n" +
                QDir::currentPath() + "/plugins_py\nor\n" +
                QDir::currentPath() + "/../plugins_py");
            return;
        }
    }

    // Scan for .py files, excluding base_plugin.py and __init__.py
    QStringList filters;
    filters << "*.py";
    dir.setNameFilters(filters);

    QStringList excludeFiles;
    excludeFiles << "base_plugin.py" << "__init__.py" << "plugin_loader.py"
                 << "test_plugins.py";

    QFileInfoList fileList = dir.entryInfoList(QDir::Files);

    int loadedCount = 0;
    int skippedCount = 0;
    QString loadedPlugins;

    for (const QFileInfo &fileInfo : fileList) {
        QString fileName = fileInfo.fileName();

        // Skip excluded files and setup files
        if (excludeFiles.contains(fileName) || fileName.startsWith("setup_")) {
            skippedCount++;
            continue;
        }

        // Check if already in list
        bool alreadyLoaded = false;
        for (int i = 0; i < pluginList->count(); ++i) {
            if (pluginList->item(i)->text().contains(fileName)) {
                alreadyLoaded = true;
                break;
            }
        }

        if (!alreadyLoaded) {
            QString displayName = fileName;
            displayName.replace(".py", "");
            displayName.replace("_", " ");

            // Capitalize first letter of each word
            QStringList words = displayName.split(" ");
            for (int i = 0; i < words.size(); ++i) {
                if (!words[i].isEmpty()) {
                    words[i][0] = words[i][0].toUpper();
                }
            }
            displayName = words.join(" ");

            pluginList->addItem(displayName + " [Python]");
            loadedPlugins += "  • " + displayName + "\n";
            loadedCount++;
        }
    }

    // Show summary
    QString message;
    if (loadedCount > 0) {
        message = QString("Successfully loaded %1 Python plugin(s):\n\n%2")
                      .arg(loadedCount)
                      .arg(loadedPlugins);
    } else {
        message = "No new Python plugins found in:\n" + pluginsDir;
    }

    if (skippedCount > 0) {
        message += QString("\n\nSkipped %1 system/base file(s)").arg(skippedCount);
    }

    pluginInfoLabel->setText(message);
    QMessageBox::information(this, "Auto-Load Complete", message);
}
