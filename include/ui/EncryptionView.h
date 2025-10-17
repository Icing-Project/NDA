#ifndef ENCRYPTIONVIEW_H
#define ENCRYPTIONVIEW_H

#include <QWidget>
#include <QLineEdit>
#include <QPushButton>
#include <QLabel>
#include <QComboBox>

class EncryptionView : public QWidget
{
    Q_OBJECT

public:
    explicit EncryptionView(QWidget *parent = nullptr);
    ~EncryptionView();

private slots:
    void onGenerateKey();
    void onImportKey();
    void onExportKey();

private:
    void setupUI();

    QComboBox *algorithmCombo;
    QLineEdit *keyDisplay;
    QPushButton *generateButton;
    QPushButton *importButton;
    QPushButton *exportButton;
    QLabel *statusLabel;
};

#endif // ENCRYPTIONVIEW_H
