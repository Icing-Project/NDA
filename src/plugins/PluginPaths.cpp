#include "plugins/PluginPaths.h"
#include <QCoreApplication>
#include <QDir>
#include <QDebug>
#include <cstdlib>

namespace nda {

QString PluginPaths::getEnvironmentPluginPath()
{
    const char* envPath = std::getenv("NDA_PLUGIN_PATH");
    if (envPath && envPath[0] != '\0') {
        QString path = QString::fromUtf8(envPath);
        qDebug() << "[PluginPaths] Using NDA_PLUGIN_PATH:" << path;
        return path;
    }
    return QString();
}

QStringList PluginPaths::getCppPluginSearchPaths()
{
    QStringList paths;
    const QString appDir = QCoreApplication::applicationDirPath();

    // Tier 1: Environment variable override
    QString envPath = getEnvironmentPluginPath();
    if (!envPath.isEmpty()) {
        paths << envPath;
    }

    // Tier 2: Application-relative paths (primary runtime)
    paths << appDir + "/plugins";              // Deployed: bin/../plugins
    paths << appDir + "/../plugins";           // Alt deployed layout

    // Tier 3: Build-tree paths (development)
    paths << appDir + "/../plugins/Release";   // MSVC multi-config (legacy)
    paths << appDir + "/../plugins/Debug";     // MSVC multi-config (legacy)
    paths << appDir + "/../build/plugins";     // Single-config builds
    paths << appDir + "/../../build/plugins";  // Deep build trees

    qDebug() << "[PluginPaths] C++ plugin search order:";
    for (const auto& path : paths) {
        qDebug() << "  -" << path << (QDir(path).exists() ? "[EXISTS]" : "[not found]");
    }

    return paths;
}

QStringList PluginPaths::getPythonPluginSearchPaths()
{
    QStringList paths;
    const QString appDir = QCoreApplication::applicationDirPath();

    // Tier 1: Environment variable override
    QString envPath = getEnvironmentPluginPath();
    if (!envPath.isEmpty()) {
        paths << envPath;
    }

    // Tier 2: Application-relative paths
    paths << appDir + "/plugins_py";           // Deployed alongside bin
    paths << appDir + "/../plugins_py";        // Alt deployed layout

    // Tier 3: Build-tree paths
    paths << appDir + "/../../plugins_py";     // Source tree from build dir

    qDebug() << "[PluginPaths] Python plugin search order:";
    for (const auto& path : paths) {
        qDebug() << "  -" << path << (QDir(path).exists() ? "[EXISTS]" : "[not found]");
    }

    return paths;
}

bool PluginPaths::validateAndLogPath(const QString& path, const QString& type)
{
    bool exists = QDir(path).exists();
    qDebug() << QString("[PluginPaths] %1 path: %2 %3")
                .arg(type)
                .arg(path)
                .arg(exists ? "[OK]" : "[MISSING]");
    return exists;
}

} // namespace nda
