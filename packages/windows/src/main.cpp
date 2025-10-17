#include <QApplication>
#include <QStyleFactory>
#include <QPalette>
#include "ui/MainWindow.h"

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);

    // Set application metadata
    app.setApplicationName("NADE");
    app.setApplicationVersion("1.0.0");
    app.setOrganizationName("NADE");

    // Set modern dark theme
    app.setStyle(QStyleFactory::create("Fusion"));
    QPalette darkPalette;

    // Modern dark color scheme - slate/blue theme
    darkPalette.setColor(QPalette::Window, QColor(15, 23, 42));           // Darker blue-gray
    darkPalette.setColor(QPalette::WindowText, QColor(241, 245, 249));    // Almost white
    darkPalette.setColor(QPalette::Base, QColor(30, 41, 59));             // Dark blue-gray
    darkPalette.setColor(QPalette::AlternateBase, QColor(51, 65, 85));    // Lighter blue-gray
    darkPalette.setColor(QPalette::ToolTipBase, QColor(30, 41, 59));
    darkPalette.setColor(QPalette::ToolTipText, QColor(241, 245, 249));
    darkPalette.setColor(QPalette::Text, QColor(241, 245, 249));          // Almost white
    darkPalette.setColor(QPalette::Button, QColor(51, 65, 85));           // Medium blue-gray
    darkPalette.setColor(QPalette::ButtonText, QColor(241, 245, 249));
    darkPalette.setColor(QPalette::BrightText, QColor(239, 68, 68));      // Red accent
    darkPalette.setColor(QPalette::Link, QColor(59, 130, 246));           // Blue link
    darkPalette.setColor(QPalette::Highlight, QColor(59, 130, 246));      // Blue highlight
    darkPalette.setColor(QPalette::HighlightedText, Qt::white);
    darkPalette.setColor(QPalette::PlaceholderText, QColor(148, 163, 184)); // Gray placeholder

    app.setPalette(darkPalette);

    // Global stylesheet for modern UI elements
    app.setStyleSheet(R"(
        QMainWindow {
            background-color: #0f172a;
        }

        QTabWidget::pane {
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 8px;
            background-color: #1e293b;
            top: -1px;
        }

        QTabBar::tab {
            background-color: #334155;
            color: #94a3b8;
            border: none;
            padding: 12px 24px;
            margin-right: 4px;
            border-top-left-radius: 8px;
            border-top-right-radius: 8px;
            font-weight: 600;
        }

        QTabBar::tab:selected {
            background-color: #1e293b;
            color: #f1f5f9;
        }

        QTabBar::tab:hover {
            background-color: #475569;
            color: #e2e8f0;
        }

        QGroupBox {
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 8px;
            margin-top: 12px;
            padding-top: 16px;
            background-color: rgba(51, 65, 85, 0.3);
            font-weight: 600;
        }

        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 0 8px;
            color: #cbd5e1;
        }

        QComboBox {
            background-color: #334155;
            border: 1px solid #475569;
            border-radius: 6px;
            padding: 8px;
            color: #f1f5f9;
            min-height: 30px;
        }

        QComboBox:hover {
            border: 1px solid #3b82f6;
        }

        QComboBox::drop-down {
            border: none;
            width: 30px;
        }

        QComboBox::down-arrow {
            image: none;
            border-left: 5px solid transparent;
            border-right: 5px solid transparent;
            border-top: 6px solid #94a3b8;
            margin-right: 8px;
        }

        QPushButton {
            background-color: #3b82f6;
            color: white;
            border: none;
            border-radius: 6px;
            padding: 10px 16px;
            font-weight: 600;
        }

        QPushButton:hover {
            background-color: #2563eb;
        }

        QPushButton:pressed {
            background-color: #1d4ed8;
        }

        QPushButton:disabled {
            background-color: #475569;
            color: #64748b;
        }

        QLabel {
            color: #e2e8f0;
        }

        QMenuBar {
            background-color: #1e293b;
            color: #f1f5f9;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }

        QMenuBar::item {
            padding: 8px 12px;
            background-color: transparent;
        }

        QMenuBar::item:selected {
            background-color: #334155;
        }

        QMenu {
            background-color: #1e293b;
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 6px;
        }

        QMenu::item {
            padding: 8px 24px;
            color: #f1f5f9;
        }

        QMenu::item:selected {
            background-color: #334155;
        }

        QStatusBar {
            background-color: #1e293b;
            color: #94a3b8;
            border-top: 1px solid rgba(255, 255, 255, 0.1);
        }

        QScrollBar:vertical {
            background-color: #1e293b;
            width: 12px;
            border-radius: 6px;
        }

        QScrollBar::handle:vertical {
            background-color: #475569;
            border-radius: 6px;
            min-height: 20px;
        }

        QScrollBar::handle:vertical:hover {
            background-color: #64748b;
        }
    )");

    // Create and show main window
    MainWindow window;
    window.show();

    return app.exec();
}
