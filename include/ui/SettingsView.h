#ifndef SETTINGSVIEW_H
#define SETTINGSVIEW_H

#include <QWidget>
#include <QCheckBox>
#include <QSpinBox>
#include <QPushButton>

class SettingsView : public QWidget
{
    Q_OBJECT

public:
    explicit SettingsView(QWidget *parent = nullptr);
    ~SettingsView();

private slots:
    void onSaveSettings();
    void onResetSettings();

private:
    void setupUI();
    void loadSettings();
    QWidget* createSettingsCard();
    void applyModernStyles();

    QCheckBox *autoStartCheck;
    QCheckBox *minimizeToTrayCheck;
    QCheckBox *hardwareAccelCheck;
    QSpinBox *maxLatencySpin;
    QSpinBox *cpuPrioritySpin;
    QPushButton *saveButton;
    QPushButton *resetButton;
};

#endif // SETTINGSVIEW_H
