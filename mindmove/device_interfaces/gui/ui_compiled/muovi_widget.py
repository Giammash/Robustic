# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'muovi_widget.ui'
##
## Created by: Qt User Interface Compiler version 6.6.0
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (
    QCoreApplication,
    QDate,
    QDateTime,
    QLocale,
    QMetaObject,
    QObject,
    QPoint,
    QRect,
    QSize,
    QTime,
    QUrl,
    Qt,
)
from PySide6.QtGui import (
    QBrush,
    QColor,
    QConicalGradient,
    QCursor,
    QFont,
    QFontDatabase,
    QGradient,
    QIcon,
    QImage,
    QKeySequence,
    QLinearGradient,
    QPainter,
    QPalette,
    QPixmap,
    QRadialGradient,
    QTransform,
)
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QGridLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSpacerItem,
    QWidget,
)


class Ui_MuoviForm(object):
    def setupUi(self, MuoviForm):
        if not MuoviForm.objectName():
            MuoviForm.setObjectName("MuoviForm")
        MuoviForm.resize(400, 324)
        self.gridLayout = QGridLayout(MuoviForm)
        self.gridLayout.setObjectName("gridLayout")
        self.verticalSpacer = QSpacerItem(
            20, 86, QSizePolicy.Minimum, QSizePolicy.Expanding
        )

        self.gridLayout.addItem(self.verticalSpacer, 4, 0, 1, 1)

        self.connectionGroupBox = QGroupBox(MuoviForm)
        self.connectionGroupBox.setObjectName("connectionGroupBox")
        self.gridLayout_7 = QGridLayout(self.connectionGroupBox)
        self.gridLayout_7.setObjectName("gridLayout_7")
        self.connectionPortLabel = QLabel(self.connectionGroupBox)
        self.connectionPortLabel.setObjectName("connectionPortLabel")

        self.gridLayout_7.addWidget(self.connectionPortLabel, 1, 1, 1, 1)

        self.label_7 = QLabel(self.connectionGroupBox)
        self.label_7.setObjectName("label_7")

        self.gridLayout_7.addWidget(self.label_7, 1, 0, 1, 1)

        self.connectionIPAddressLabel = QLabel(self.connectionGroupBox)
        self.connectionIPAddressLabel.setObjectName("connectionIPAddressLabel")

        self.gridLayout_7.addWidget(self.connectionIPAddressLabel, 0, 1, 1, 1)

        self.label_6 = QLabel(self.connectionGroupBox)
        self.label_6.setObjectName("label_6")

        self.gridLayout_7.addWidget(self.label_6, 0, 0, 1, 1)

        self.connectionUpdatePushButton = QPushButton(self.connectionGroupBox)
        self.connectionUpdatePushButton.setObjectName("connectionUpdatePushButton")

        self.gridLayout_7.addWidget(self.connectionUpdatePushButton, 0, 2, 1, 1)

        self.gridLayout.addWidget(self.connectionGroupBox, 0, 0, 1, 2)

        self.commandsGroupBox = QGroupBox(MuoviForm)
        self.commandsGroupBox.setObjectName("commandsGroupBox")
        self.gridLayout_3 = QGridLayout(self.commandsGroupBox)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.commandConnectionPushButton = QPushButton(self.commandsGroupBox)
        self.commandConnectionPushButton.setObjectName("commandConnectionPushButton")

        self.gridLayout_3.addWidget(self.commandConnectionPushButton, 0, 0, 1, 1)

        self.commandConfigurationPushButton = QPushButton(self.commandsGroupBox)
        self.commandConfigurationPushButton.setObjectName(
            "commandConfigurationPushButton"
        )

        self.gridLayout_3.addWidget(self.commandConfigurationPushButton, 1, 0, 1, 1)

        self.commandStreamPushButton = QPushButton(self.commandsGroupBox)
        self.commandStreamPushButton.setObjectName("commandStreamPushButton")

        self.gridLayout_3.addWidget(self.commandStreamPushButton, 2, 0, 1, 1)

        self.gridLayout.addWidget(self.commandsGroupBox, 2, 0, 2, 2)

        self.inputGroupBox = QGroupBox(MuoviForm)
        self.inputGroupBox.setObjectName("inputGroupBox")
        self.gridLayout_4 = QGridLayout(self.inputGroupBox)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.inputDetectionModeComboBox = QComboBox(self.inputGroupBox)
        self.inputDetectionModeComboBox.addItem("")
        self.inputDetectionModeComboBox.addItem("")
        self.inputDetectionModeComboBox.addItem("")
        self.inputDetectionModeComboBox.addItem("")
        self.inputDetectionModeComboBox.setObjectName("inputDetectionModeComboBox")

        self.gridLayout_4.addWidget(self.inputDetectionModeComboBox, 1, 1, 1, 1)

        self.label_10 = QLabel(self.inputGroupBox)
        self.label_10.setObjectName("label_10")

        self.gridLayout_4.addWidget(self.label_10, 1, 0, 1, 1)

        self.label = QLabel(self.inputGroupBox)
        self.label.setObjectName("label")

        self.gridLayout_4.addWidget(self.label, 0, 0, 1, 1)

        self.inputWorkingModeComboBox = QComboBox(self.inputGroupBox)
        self.inputWorkingModeComboBox.addItem("")
        self.inputWorkingModeComboBox.addItem("")
        self.inputWorkingModeComboBox.setObjectName("inputWorkingModeComboBox")

        self.gridLayout_4.addWidget(self.inputWorkingModeComboBox, 0, 1, 1, 1)

        self.gridLayout.addWidget(self.inputGroupBox, 1, 0, 1, 2)

        self.retranslateUi(MuoviForm)

        self.inputDetectionModeComboBox.setCurrentIndex(1)
        self.inputWorkingModeComboBox.setCurrentIndex(1)

        QMetaObject.connectSlotsByName(MuoviForm)

    # setupUi

    def retranslateUi(self, MuoviForm):
        MuoviForm.setWindowTitle(
            QCoreApplication.translate("MuoviForm", "MuoviForm", None)
        )
        self.connectionGroupBox.setTitle(
            QCoreApplication.translate("MuoviForm", "Connection parameters", None)
        )
        self.connectionPortLabel.setText(
            QCoreApplication.translate("MuoviForm", "54321", None)
        )
        self.label_7.setText(QCoreApplication.translate("MuoviForm", "Port", None))
        self.connectionIPAddressLabel.setText(
            QCoreApplication.translate("MuoviForm", "Placeholder", None)
        )
        self.label_6.setText(QCoreApplication.translate("MuoviForm", "IP", None))
        self.connectionUpdatePushButton.setText(
            QCoreApplication.translate("MuoviForm", "Update", None)
        )
        self.commandsGroupBox.setTitle(
            QCoreApplication.translate("MuoviForm", "Commands", None)
        )
        self.commandConnectionPushButton.setText(
            QCoreApplication.translate("MuoviForm", "Connect", None)
        )
        self.commandConfigurationPushButton.setText(
            QCoreApplication.translate("MuoviForm", "Configure", None)
        )
        self.commandStreamPushButton.setText(
            QCoreApplication.translate("MuoviForm", "Stream", None)
        )
        self.inputGroupBox.setTitle(
            QCoreApplication.translate("MuoviForm", "Input Parameters", None)
        )
        self.inputDetectionModeComboBox.setItemText(
            0, QCoreApplication.translate("MuoviForm", "Monopolar - High Gain", None)
        )
        self.inputDetectionModeComboBox.setItemText(
            1, QCoreApplication.translate("MuoviForm", "Monopolar - Low Gain", None)
        )
        self.inputDetectionModeComboBox.setItemText(
            2, QCoreApplication.translate("MuoviForm", "Impedance Check", None)
        )
        self.inputDetectionModeComboBox.setItemText(
            3, QCoreApplication.translate("MuoviForm", "Test", None)
        )

        self.label_10.setText(
            QCoreApplication.translate("MuoviForm", "Detection Mode", None)
        )
        self.label.setText(
            QCoreApplication.translate("MuoviForm", "Working Mode", None)
        )
        self.inputWorkingModeComboBox.setItemText(
            0, QCoreApplication.translate("MuoviForm", "EEG", None)
        )
        self.inputWorkingModeComboBox.setItemText(
            1, QCoreApplication.translate("MuoviForm", "EMG", None)
        )

    # retranslateUi
