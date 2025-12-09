# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'main_window.ui'
##
## Created by: Qt User Interface Compiler version 6.7.1
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
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QGridLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenuBar,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QSpacerItem,
    QSpinBox,
    QStackedWidget,
    QStatusBar,
    QTabWidget,
    QWidget,
)

from mindmove.device_interfaces.gui.muovi_widget import MuoviWidget
# from mindmove.device_interfaces.gui.muovi_widget import MuoviWidget
from mindmove.gui_custom_elements.vispy_plot_widget import VispyPlotWidget


class Ui_MindMove(object):
    def setupUi(self, MindMove):
        if not MindMove.objectName():
            MindMove.setObjectName("MindMove")
        MindMove.resize(804, 607)
        self.centralwidget = QWidget(MindMove)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.vispyPlotWidget = VispyPlotWidget(self.centralwidget)
        self.vispyPlotWidget.setObjectName("vispyPlotWidget")
        sizePolicy = QSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.vispyPlotWidget.sizePolicy().hasHeightForWidth()
        )
        self.vispyPlotWidget.setSizePolicy(sizePolicy)
        self.gridLayout_6 = QGridLayout(self.vispyPlotWidget)
        self.gridLayout_6.setObjectName("gridLayout_6")
        self.vispyPlotEnabledCheckBox = QCheckBox(self.vispyPlotWidget)
        self.vispyPlotEnabledCheckBox.setObjectName("vispyPlotEnabledCheckBox")

        self.gridLayout_6.addWidget(self.vispyPlotEnabledCheckBox, 0, 0, 1, 1)

        self.verticalSpacer_3 = QSpacerItem(
            20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding
        )

        self.gridLayout_6.addItem(self.verticalSpacer_3, 1, 0, 1, 1)

        self.horizontalSpacer = QSpacerItem(
            40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum
        )

        self.gridLayout_6.addItem(self.horizontalSpacer, 0, 1, 1, 1)

        self.gridLayout.addWidget(self.vispyPlotWidget, 0, 1, 1, 1)

        self.mindMoveTabWidget = QTabWidget(self.centralwidget)
        self.mindMoveTabWidget.setObjectName("mindMoveTabWidget")
        sizePolicy1 = QSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding
        )
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(
            self.mindMoveTabWidget.sizePolicy().hasHeightForWidth()
        )
        self.mindMoveTabWidget.setSizePolicy(sizePolicy1)
        self.muoviWidget = MuoviWidget()
        self.muoviWidget.setObjectName("muoviWidget")
        self.mindMoveTabWidget.addTab(self.muoviWidget, "")
        self.virtualHandWidget = QWidget()
        self.virtualHandWidget.setObjectName("virtualHandWidget")
        self.gridLayout_4 = QGridLayout(self.virtualHandWidget)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.verticalSpacer_4 = QSpacerItem(
            20, 99, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding
        )

        self.gridLayout_4.addItem(self.verticalSpacer_4, 3, 1, 1, 1)

        self.mindmoveUDPSocketGroupBox = QGroupBox(self.virtualHandWidget)
        self.mindmoveUDPSocketGroupBox.setObjectName("mindmoveUDPSocketGroupBox")
        self.gridLayout_15 = QGridLayout(self.mindmoveUDPSocketGroupBox)
        self.gridLayout_15.setObjectName("gridLayout_15")
        self.label_42 = QLabel(self.mindmoveUDPSocketGroupBox)
        self.label_42.setObjectName("label_42")

        self.gridLayout_15.addWidget(self.label_42, 3, 0, 1, 1)

        self.mindmoveUDPPortLineEdit = QLineEdit(self.mindmoveUDPSocketGroupBox)
        self.mindmoveUDPPortLineEdit.setObjectName("mindmoveUDPPortLineEdit")
        sizePolicy2 = QSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed
        )
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(
            self.mindmoveUDPPortLineEdit.sizePolicy().hasHeightForWidth()
        )
        self.mindmoveUDPPortLineEdit.setSizePolicy(sizePolicy2)

        self.gridLayout_15.addWidget(self.mindmoveUDPPortLineEdit, 1, 1, 1, 1)

        self.label_19 = QLabel(self.mindmoveUDPSocketGroupBox)
        self.label_19.setObjectName("label_19")

        self.gridLayout_15.addWidget(self.label_19, 1, 0, 1, 1)

        self.mindmoveStreamingFrequencyComboBox = QComboBox(
            self.mindmoveUDPSocketGroupBox
        )
        self.mindmoveStreamingFrequencyComboBox.addItem("")
        self.mindmoveStreamingFrequencyComboBox.addItem("")
        self.mindmoveStreamingFrequencyComboBox.addItem("")
        self.mindmoveStreamingFrequencyComboBox.addItem("")
        self.mindmoveStreamingFrequencyComboBox.addItem("")
        self.mindmoveStreamingFrequencyComboBox.addItem("")
        self.mindmoveStreamingFrequencyComboBox.setObjectName(
            "mindmoveStreamingFrequencyComboBox"
        )

        self.gridLayout_15.addWidget(
            self.mindmoveStreamingFrequencyComboBox, 3, 1, 1, 1
        )

        self.mindmoveUDPIPLineEdit = QLineEdit(self.mindmoveUDPSocketGroupBox)
        self.mindmoveUDPIPLineEdit.setObjectName("mindmoveUDPIPLineEdit")
        sizePolicy2.setHeightForWidth(
            self.mindmoveUDPIPLineEdit.sizePolicy().hasHeightForWidth()
        )
        self.mindmoveUDPIPLineEdit.setSizePolicy(sizePolicy2)

        self.gridLayout_15.addWidget(self.mindmoveUDPIPLineEdit, 0, 1, 1, 1)

        self.label_18 = QLabel(self.mindmoveUDPSocketGroupBox)
        self.label_18.setObjectName("label_18")

        self.gridLayout_15.addWidget(self.label_18, 0, 0, 1, 1)

        self.gridLayout_4.addWidget(self.mindmoveUDPSocketGroupBox, 0, 0, 1, 2)

        self.virtualHandInterfaceGroupBox = QGroupBox(self.virtualHandWidget)
        self.virtualHandInterfaceGroupBox.setObjectName("virtualHandInterfaceGroupBox")
        self.gridLayout_7 = QGridLayout(self.virtualHandInterfaceGroupBox)
        self.gridLayout_7.setObjectName("gridLayout_7")
        self.virtualHandInterfaceIPLineEdit = QLineEdit(
            self.virtualHandInterfaceGroupBox
        )
        self.virtualHandInterfaceIPLineEdit.setObjectName(
            "virtualHandInterfaceIPLineEdit"
        )
        sizePolicy2.setHeightForWidth(
            self.virtualHandInterfaceIPLineEdit.sizePolicy().hasHeightForWidth()
        )
        self.virtualHandInterfaceIPLineEdit.setSizePolicy(sizePolicy2)

        self.gridLayout_7.addWidget(self.virtualHandInterfaceIPLineEdit, 0, 1, 1, 1)

        self.label_3 = QLabel(self.virtualHandInterfaceGroupBox)
        self.label_3.setObjectName("label_3")

        self.gridLayout_7.addWidget(self.label_3, 0, 0, 1, 1)

        self.label_4 = QLabel(self.virtualHandInterfaceGroupBox)
        self.label_4.setObjectName("label_4")

        self.gridLayout_7.addWidget(self.label_4, 1, 0, 1, 1)

        self.virtualHandInterfacePortLineEdit = QLineEdit(
            self.virtualHandInterfaceGroupBox
        )
        self.virtualHandInterfacePortLineEdit.setObjectName(
            "virtualHandInterfacePortLineEdit"
        )
        sizePolicy2.setHeightForWidth(
            self.virtualHandInterfacePortLineEdit.sizePolicy().hasHeightForWidth()
        )
        self.virtualHandInterfacePortLineEdit.setSizePolicy(sizePolicy2)

        self.gridLayout_7.addWidget(self.virtualHandInterfacePortLineEdit, 1, 1, 1, 1)

        self.gridLayout_4.addWidget(self.virtualHandInterfaceGroupBox, 1, 0, 1, 2)

        self.groupBox_2 = QGroupBox(self.virtualHandWidget)
        self.groupBox_2.setObjectName("groupBox_2")
        self.gridLayout_8 = QGridLayout(self.groupBox_2)
        self.gridLayout_8.setObjectName("gridLayout_8")
        self.configureAppUdpSocketPushButton = QPushButton(self.groupBox_2)
        self.configureAppUdpSocketPushButton.setObjectName(
            "configureAppUdpSocketPushButton"
        )
        self.configureAppUdpSocketPushButton.setCheckable(True)

        self.gridLayout_8.addWidget(self.configureAppUdpSocketPushButton, 0, 0, 1, 1)

        self.outputToggleStreamingPushButton = QPushButton(self.groupBox_2)
        self.outputToggleStreamingPushButton.setObjectName(
            "outputToggleStreamingPushButton"
        )
        self.outputToggleStreamingPushButton.setToolTipDuration(5)
        self.outputToggleStreamingPushButton.setCheckable(True)
        self.outputToggleStreamingPushButton.setChecked(False)

        self.gridLayout_8.addWidget(self.outputToggleStreamingPushButton, 0, 1, 1, 1)

        self.gridLayout_4.addWidget(self.groupBox_2, 2, 0, 1, 2)

        self.mindMoveTabWidget.addTab(self.virtualHandWidget, "")
        self.procotolWidget = QWidget()
        self.procotolWidget.setObjectName("procotolWidget")
        self.gridLayout_2 = QGridLayout(self.procotolWidget)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.protocolModeStackedWidget = QStackedWidget(self.procotolWidget)
        self.protocolModeStackedWidget.setObjectName("protocolModeStackedWidget")
        self.recordWidget = QWidget()
        self.recordWidget.setObjectName("recordWidget")
        self.gridLayout_5 = QGridLayout(self.recordWidget)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.recordReviewRecordingStackedWidget = QStackedWidget(self.recordWidget)
        self.recordReviewRecordingStackedWidget.setObjectName(
            "recordReviewRecordingStackedWidget"
        )
        self.emptyWidget_2 = QWidget()
        self.emptyWidget_2.setObjectName("emptyWidget_2")
        self.recordReviewRecordingStackedWidget.addWidget(self.emptyWidget_2)
        self.reviewRecordingWidget = QWidget()
        self.reviewRecordingWidget.setObjectName("reviewRecordingWidget")
        self.gridLayout_11 = QGridLayout(self.reviewRecordingWidget)
        self.gridLayout_11.setObjectName("gridLayout_11")
        self.reviewRecordingGroupBox = QGroupBox(self.reviewRecordingWidget)
        self.reviewRecordingGroupBox.setObjectName("reviewRecordingGroupBox")
        sizePolicy1.setHeightForWidth(
            self.reviewRecordingGroupBox.sizePolicy().hasHeightForWidth()
        )
        self.reviewRecordingGroupBox.setSizePolicy(sizePolicy1)
        self.gridLayout_10 = QGridLayout(self.reviewRecordingGroupBox)
        self.gridLayout_10.setObjectName("gridLayout_10")
        self.label_5 = QLabel(self.reviewRecordingGroupBox)
        self.label_5.setObjectName("label_5")

        self.gridLayout_10.addWidget(self.label_5, 0, 0, 1, 1)

        self.reviewRecordingTaskLabel = QLabel(self.reviewRecordingGroupBox)
        self.reviewRecordingTaskLabel.setObjectName("reviewRecordingTaskLabel")

        self.gridLayout_10.addWidget(self.reviewRecordingTaskLabel, 0, 1, 1, 1)

        self.label_2 = QLabel(self.reviewRecordingGroupBox)
        self.label_2.setObjectName("label_2")

        self.gridLayout_10.addWidget(self.label_2, 1, 0, 1, 1)

        self.reviewRecordingKinematicsPlotWidget = VispyPlotWidget(
            self.reviewRecordingGroupBox
        )
        self.reviewRecordingKinematicsPlotWidget.setObjectName(
            "reviewRecordingKinematicsPlotWidget"
        )
        sizePolicy1.setHeightForWidth(
            self.reviewRecordingKinematicsPlotWidget.sizePolicy().hasHeightForWidth()
        )
        self.reviewRecordingKinematicsPlotWidget.setSizePolicy(sizePolicy1)

        self.gridLayout_10.addWidget(
            self.reviewRecordingKinematicsPlotWidget, 4, 0, 1, 2
        )

        self.reviewRecordingLabelLineEdit = QLineEdit(self.reviewRecordingGroupBox)
        self.reviewRecordingLabelLineEdit.setObjectName("reviewRecordingLabelLineEdit")
        sizePolicy3 = QSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred
        )
        sizePolicy3.setHorizontalStretch(0)
        sizePolicy3.setVerticalStretch(0)
        sizePolicy3.setHeightForWidth(
            self.reviewRecordingLabelLineEdit.sizePolicy().hasHeightForWidth()
        )
        self.reviewRecordingLabelLineEdit.setSizePolicy(sizePolicy3)

        self.gridLayout_10.addWidget(self.reviewRecordingLabelLineEdit, 1, 1, 1, 1)

        self.reviewRecordingAcceptPushButton = QPushButton(self.reviewRecordingGroupBox)
        self.reviewRecordingAcceptPushButton.setObjectName(
            "reviewRecordingAcceptPushButton"
        )
        sizePolicy4 = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        sizePolicy4.setHorizontalStretch(0)
        sizePolicy4.setVerticalStretch(0)
        sizePolicy4.setHeightForWidth(
            self.reviewRecordingAcceptPushButton.sizePolicy().hasHeightForWidth()
        )
        self.reviewRecordingAcceptPushButton.setSizePolicy(sizePolicy4)
        self.reviewRecordingAcceptPushButton.setStyleSheet(
            "color: rgb(0, 0, 0); background-color: rgb(170, 255, 0);"
        )

        self.gridLayout_10.addWidget(self.reviewRecordingAcceptPushButton, 5, 0, 1, 1)

        self.reviewRecordingEMGPlotWidget = VispyPlotWidget(
            self.reviewRecordingGroupBox
        )
        self.reviewRecordingEMGPlotWidget.setObjectName("reviewRecordingEMGPlotWidget")
        sizePolicy1.setHeightForWidth(
            self.reviewRecordingEMGPlotWidget.sizePolicy().hasHeightForWidth()
        )
        self.reviewRecordingEMGPlotWidget.setSizePolicy(sizePolicy1)

        self.gridLayout_10.addWidget(self.reviewRecordingEMGPlotWidget, 3, 0, 1, 2)

        self.reviewRecordingRejectPushButton = QPushButton(self.reviewRecordingGroupBox)
        self.reviewRecordingRejectPushButton.setObjectName(
            "reviewRecordingRejectPushButton"
        )
        sizePolicy4.setHeightForWidth(
            self.reviewRecordingRejectPushButton.sizePolicy().hasHeightForWidth()
        )
        self.reviewRecordingRejectPushButton.setSizePolicy(sizePolicy4)
        self.reviewRecordingRejectPushButton.setLayoutDirection(
            Qt.LayoutDirection.LeftToRight
        )
        self.reviewRecordingRejectPushButton.setStyleSheet(
            "background-color: rgb(255, 0, 0);\ncolor: rgb(0, 0, 0);"
        )

        self.gridLayout_10.addWidget(self.reviewRecordingRejectPushButton, 5, 1, 1, 1)

        self.gridLayout_11.addWidget(self.reviewRecordingGroupBox, 0, 0, 1, 1)

        self.recordReviewRecordingStackedWidget.addWidget(self.reviewRecordingWidget)

        self.gridLayout_5.addWidget(self.recordReviewRecordingStackedWidget, 1, 0, 1, 1)

        self.recordRecordingGroupBox = QGroupBox(self.recordWidget)
        self.recordRecordingGroupBox.setObjectName("recordRecordingGroupBox")
        self.gridLayout_9 = QGridLayout(self.recordRecordingGroupBox)
        self.gridLayout_9.setObjectName("gridLayout_9")
        self.label = QLabel(self.recordRecordingGroupBox)
        self.label.setObjectName("label")

        self.gridLayout_9.addWidget(self.label, 0, 0, 1, 1)

        self.recordTaskComboBox = QComboBox(self.recordRecordingGroupBox)
        self.recordTaskComboBox.addItem("")
        self.recordTaskComboBox.addItem("")
        self.recordTaskComboBox.addItem("")
        self.recordTaskComboBox.addItem("")
        self.recordTaskComboBox.addItem("")
        self.recordTaskComboBox.addItem("")
        self.recordTaskComboBox.addItem("")
        self.recordTaskComboBox.addItem("")
        self.recordTaskComboBox.addItem("")
        self.recordTaskComboBox.setObjectName("recordTaskComboBox")

        self.gridLayout_9.addWidget(self.recordTaskComboBox, 0, 1, 1, 1)

        self.recordRecordPushButton = QPushButton(self.recordRecordingGroupBox)
        self.recordRecordPushButton.setObjectName("recordRecordPushButton")
        self.recordRecordPushButton.setCheckable(True)

        self.gridLayout_9.addWidget(self.recordRecordPushButton, 2, 0, 1, 2)

        self.recordEMGProgressBar = QProgressBar(self.recordRecordingGroupBox)
        self.recordEMGProgressBar.setObjectName("recordEMGProgressBar")
        self.recordEMGProgressBar.setValue(24)

        self.gridLayout_9.addWidget(self.recordEMGProgressBar, 3, 0, 1, 1)

        self.recordKinematicsProgressBar = QProgressBar(self.recordRecordingGroupBox)
        self.recordKinematicsProgressBar.setObjectName("recordKinematicsProgressBar")
        self.recordKinematicsProgressBar.setValue(24)

        self.gridLayout_9.addWidget(self.recordKinematicsProgressBar, 3, 1, 1, 1)

        self.label_7 = QLabel(self.recordRecordingGroupBox)
        self.label_7.setObjectName("label_7")

        self.gridLayout_9.addWidget(self.label_7, 1, 0, 1, 1)

        self.recordDurationSpinBox = QSpinBox(self.recordRecordingGroupBox)
        self.recordDurationSpinBox.setObjectName("recordDurationSpinBox")
        self.recordDurationSpinBox.setValue(10)

        self.gridLayout_9.addWidget(self.recordDurationSpinBox, 1, 1, 1, 1)

        self.gridLayout_5.addWidget(self.recordRecordingGroupBox, 0, 0, 1, 2)

        self.protocolModeStackedWidget.addWidget(self.recordWidget)
        self.trainingWidget = QWidget()
        self.trainingWidget.setObjectName("trainingWidget")
        self.gridLayout_12 = QGridLayout(self.trainingWidget)
        self.gridLayout_12.setObjectName("gridLayout_12")
        self.trainingCreateDatasetGroupBox = QGroupBox(self.trainingWidget)
        self.trainingCreateDatasetGroupBox.setObjectName(
            "trainingCreateDatasetGroupBox"
        )
        self.gridLayout_13 = QGridLayout(self.trainingCreateDatasetGroupBox)
        self.gridLayout_13.setObjectName("gridLayout_13")
        self.trainingCreateDatasetsSelectRecordingsPushButton = QPushButton(
            self.trainingCreateDatasetGroupBox
        )
        self.trainingCreateDatasetsSelectRecordingsPushButton.setObjectName(
            "trainingCreateDatasetsSelectRecordingsPushButton"
        )

        self.gridLayout_13.addWidget(
            self.trainingCreateDatasetsSelectRecordingsPushButton, 0, 0, 1, 1
        )

        self.trainingCreateDatasetPushButton = QPushButton(
            self.trainingCreateDatasetGroupBox
        )
        self.trainingCreateDatasetPushButton.setObjectName(
            "trainingCreateDatasetPushButton"
        )

        self.gridLayout_13.addWidget(self.trainingCreateDatasetPushButton, 3, 0, 1, 1)

        self.label_6 = QLabel(self.trainingCreateDatasetGroupBox)
        self.label_6.setObjectName("label_6")

        self.gridLayout_13.addWidget(self.label_6, 2, 0, 1, 1)

        self.trainingCreateDatasetSelectedRecordingsListWidget = QListWidget(
            self.trainingCreateDatasetGroupBox
        )
        self.trainingCreateDatasetSelectedRecordingsListWidget.setObjectName(
            "trainingCreateDatasetSelectedRecordingsListWidget"
        )
        self.trainingCreateDatasetSelectedRecordingsListWidget.setSelectionMode(
            QAbstractItemView.SelectionMode.NoSelection
        )

        self.gridLayout_13.addWidget(
            self.trainingCreateDatasetSelectedRecordingsListWidget, 1, 0, 1, 2
        )

        self.trainingCreateDatasetLabelLineEdit = QLineEdit(
            self.trainingCreateDatasetGroupBox
        )
        self.trainingCreateDatasetLabelLineEdit.setObjectName(
            "trainingCreateDatasetLabelLineEdit"
        )

        self.gridLayout_13.addWidget(
            self.trainingCreateDatasetLabelLineEdit, 2, 1, 1, 1
        )

        self.trainingCreateDatasetProgressBar = QProgressBar(
            self.trainingCreateDatasetGroupBox
        )
        self.trainingCreateDatasetProgressBar.setObjectName(
            "trainingCreateDatasetProgressBar"
        )
        self.trainingCreateDatasetProgressBar.setValue(24)

        self.gridLayout_13.addWidget(self.trainingCreateDatasetProgressBar, 3, 1, 1, 1)

        self.gridLayout_12.addWidget(self.trainingCreateDatasetGroupBox, 0, 0, 1, 1)

        self.trainingTrainModelGroupBox = QGroupBox(self.trainingWidget)
        self.trainingTrainModelGroupBox.setObjectName("trainingTrainModelGroupBox")
        self.gridLayout_14 = QGridLayout(self.trainingTrainModelGroupBox)
        self.gridLayout_14.setObjectName("gridLayout_14")
        self.trainingTrainNewModelRadioButton = QRadioButton(
            self.trainingTrainModelGroupBox
        )
        self.trainingTrainNewModelRadioButton.setObjectName(
            "trainingTrainNewModelRadioButton"
        )
        self.trainingTrainNewModelRadioButton.setChecked(True)

        self.gridLayout_14.addWidget(self.trainingTrainNewModelRadioButton, 0, 1, 1, 1)

        self.trainingTrainModelPushButton = QPushButton(self.trainingTrainModelGroupBox)
        self.trainingTrainModelPushButton.setObjectName("trainingTrainModelPushButton")

        self.gridLayout_14.addWidget(self.trainingTrainModelPushButton, 4, 1, 1, 1)

        self.trainingSelectDatasetPushButton = QPushButton(
            self.trainingTrainModelGroupBox
        )
        self.trainingSelectDatasetPushButton.setObjectName(
            "trainingSelectDatasetPushButton"
        )

        self.gridLayout_14.addWidget(self.trainingSelectDatasetPushButton, 2, 1, 1, 1)

        self.trainingProgressBar = QProgressBar(self.trainingTrainModelGroupBox)
        self.trainingProgressBar.setObjectName("trainingProgressBar")
        self.trainingProgressBar.setValue(24)

        self.gridLayout_14.addWidget(self.trainingProgressBar, 4, 2, 1, 1)

        self.trainingSelectedDatasetLabel = QLabel(self.trainingTrainModelGroupBox)
        self.trainingSelectedDatasetLabel.setObjectName("trainingSelectedDatasetLabel")

        self.gridLayout_14.addWidget(self.trainingSelectedDatasetLabel, 2, 2, 1, 1)

        self.trainingModelLabelLineEdit = QLineEdit(self.trainingTrainModelGroupBox)
        self.trainingModelLabelLineEdit.setObjectName("trainingModelLabelLineEdit")

        self.gridLayout_14.addWidget(self.trainingModelLabelLineEdit, 3, 2, 1, 1)

        self.trainingTrainModelStackedWidget = QStackedWidget(
            self.trainingTrainModelGroupBox
        )
        self.trainingTrainModelStackedWidget.setObjectName(
            "trainingTrainModelStackedWidget"
        )
        sizePolicy2.setHeightForWidth(
            self.trainingTrainModelStackedWidget.sizePolicy().hasHeightForWidth()
        )
        self.trainingTrainModelStackedWidget.setSizePolicy(sizePolicy2)
        self.emptyWidget = QWidget()
        self.emptyWidget.setObjectName("emptyWidget")
        self.trainingTrainModelStackedWidget.addWidget(self.emptyWidget)
        self.trainingLoadExistingModelWidget = QWidget()
        self.trainingLoadExistingModelWidget.setObjectName(
            "trainingLoadExistingModelWidget"
        )
        self.gridLayout_17 = QGridLayout(self.trainingLoadExistingModelWidget)
        self.gridLayout_17.setObjectName("gridLayout_17")
        self.trainingLoadExistingModelPushButton = QPushButton(
            self.trainingLoadExistingModelWidget
        )
        self.trainingLoadExistingModelPushButton.setObjectName(
            "trainingLoadExistingModelPushButton"
        )

        self.gridLayout_17.addWidget(
            self.trainingLoadExistingModelPushButton, 0, 0, 1, 1
        )

        self.trainingLoadExistingModelLabel = QLabel(
            self.trainingLoadExistingModelWidget
        )
        self.trainingLoadExistingModelLabel.setObjectName(
            "trainingLoadExistingModelLabel"
        )

        self.gridLayout_17.addWidget(self.trainingLoadExistingModelLabel, 0, 1, 1, 1)

        self.trainingTrainModelStackedWidget.addWidget(
            self.trainingLoadExistingModelWidget
        )

        self.gridLayout_14.addWidget(self.trainingTrainModelStackedWidget, 1, 1, 1, 2)

        self.label_8 = QLabel(self.trainingTrainModelGroupBox)
        self.label_8.setObjectName("label_8")

        self.gridLayout_14.addWidget(self.label_8, 3, 1, 1, 1)

        self.trainingTrainExistingModelRadioButton = QRadioButton(
            self.trainingTrainModelGroupBox
        )
        self.trainingTrainExistingModelRadioButton.setObjectName(
            "trainingTrainExistingModelRadioButton"
        )

        self.gridLayout_14.addWidget(
            self.trainingTrainExistingModelRadioButton, 0, 2, 1, 1
        )

        self.gridLayout_12.addWidget(self.trainingTrainModelGroupBox, 1, 0, 1, 1)

        self.protocolModeStackedWidget.addWidget(self.trainingWidget)
        self.onlineWidget = QWidget()
        self.onlineWidget.setObjectName("onlineWidget")
        self.gridLayout_16 = QGridLayout(self.onlineWidget)
        self.gridLayout_16.setObjectName("gridLayout_16")
        self.onlineLoadModelGroupBox = QGroupBox(self.onlineWidget)
        self.onlineLoadModelGroupBox.setObjectName("onlineLoadModelGroupBox")
        self.gridLayout_18 = QGridLayout(self.onlineLoadModelGroupBox)
        self.gridLayout_18.setObjectName("gridLayout_18")
        self.onlineLoadModelPushButton = QPushButton(self.onlineLoadModelGroupBox)
        self.onlineLoadModelPushButton.setObjectName("onlineLoadModelPushButton")
        self.onlineLoadModelPushButton.setCheckable(False)

        self.gridLayout_18.addWidget(self.onlineLoadModelPushButton, 0, 0, 1, 1)

        self.onlineModelLabel = QLabel(self.onlineLoadModelGroupBox)
        self.onlineModelLabel.setObjectName("onlineModelLabel")

        self.gridLayout_18.addWidget(self.onlineModelLabel, 0, 1, 1, 1)

        self.gridLayout_16.addWidget(self.onlineLoadModelGroupBox, 0, 0, 1, 2)

        self.onlineCommandsGroupBox = QGroupBox(self.onlineWidget)
        self.onlineCommandsGroupBox.setObjectName("onlineCommandsGroupBox")
        self.gridLayout_19 = QGridLayout(self.onlineCommandsGroupBox)
        self.gridLayout_19.setObjectName("gridLayout_19")
        self.onlineRecordTogglePushButton = QPushButton(self.onlineCommandsGroupBox)
        self.onlineRecordTogglePushButton.setObjectName("onlineRecordTogglePushButton")
        self.onlineRecordTogglePushButton.setCheckable(True)

        self.gridLayout_19.addWidget(self.onlineRecordTogglePushButton, 0, 0, 1, 1)

        self.gridLayout_16.addWidget(self.onlineCommandsGroupBox, 1, 0, 1, 2)

        self.verticalSpacer = QSpacerItem(
            20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding
        )

        self.gridLayout_16.addItem(self.verticalSpacer, 2, 0, 1, 1)

        self.protocolModeStackedWidget.addWidget(self.onlineWidget)

        self.gridLayout_2.addWidget(self.protocolModeStackedWidget, 1, 0, 1, 1)

        self.protocolModeGroupBox = QGroupBox(self.procotolWidget)
        self.protocolModeGroupBox.setObjectName("protocolModeGroupBox")
        self.gridLayout_3 = QGridLayout(self.protocolModeGroupBox)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.protocolTrainingRadioButton = QRadioButton(self.protocolModeGroupBox)
        self.protocolTrainingRadioButton.setObjectName("protocolTrainingRadioButton")

        self.gridLayout_3.addWidget(self.protocolTrainingRadioButton, 0, 1, 1, 1)

        self.protocolOnlineRadioButton = QRadioButton(self.protocolModeGroupBox)
        self.protocolOnlineRadioButton.setObjectName("protocolOnlineRadioButton")

        self.gridLayout_3.addWidget(self.protocolOnlineRadioButton, 0, 2, 1, 1)

        self.protocolRecordRadioButton = QRadioButton(self.protocolModeGroupBox)
        self.protocolRecordRadioButton.setObjectName("protocolRecordRadioButton")
        self.protocolRecordRadioButton.setChecked(True)

        self.gridLayout_3.addWidget(self.protocolRecordRadioButton, 0, 0, 1, 1)

        self.gridLayout_2.addWidget(self.protocolModeGroupBox, 0, 0, 1, 1)

        self.verticalSpacer_2 = QSpacerItem(
            20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding
        )

        self.gridLayout_2.addItem(self.verticalSpacer_2, 2, 0, 1, 1)

        self.mindMoveTabWidget.addTab(self.procotolWidget, "")

        self.gridLayout.addWidget(self.mindMoveTabWidget, 0, 0, 1, 1)

        MindMove.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MindMove)
        self.menubar.setObjectName("menubar")
        self.menubar.setGeometry(QRect(0, 0, 804, 33))
        MindMove.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MindMove)
        self.statusbar.setObjectName("statusbar")
        MindMove.setStatusBar(self.statusbar)

        self.retranslateUi(MindMove)

        self.mindMoveTabWidget.setCurrentIndex(1)
        self.protocolModeStackedWidget.setCurrentIndex(2)
        self.recordReviewRecordingStackedWidget.setCurrentIndex(1)
        self.trainingTrainModelStackedWidget.setCurrentIndex(1)

        QMetaObject.connectSlotsByName(MindMove)

    # setupUi

    def retranslateUi(self, MindMove):
        MindMove.setWindowTitle(
            QCoreApplication.translate("MindMove", "MindMove", None)
        )
        self.vispyPlotEnabledCheckBox.setText(
            QCoreApplication.translate("MindMove", "Plot Enabled", None)
        )
        self.mindMoveTabWidget.setTabText(
            self.mindMoveTabWidget.indexOf(self.muoviWidget),
            QCoreApplication.translate("MindMove", "Devices", None),
        )
        self.mindmoveUDPSocketGroupBox.setTitle(
            QCoreApplication.translate("MindMove", "MindMove UDP Socket Settings", None)
        )
        self.label_42.setText(
            QCoreApplication.translate("MindMove", "Streaming Frequency", None)
        )
        self.mindmoveUDPPortLineEdit.setText(
            QCoreApplication.translate("MindMove", "1233", None)
        )
        self.label_19.setText(QCoreApplication.translate("MindMove", "Port", None))
        self.mindmoveStreamingFrequencyComboBox.setItemText(
            0, QCoreApplication.translate("MindMove", "32 Hz", None)
        )
        self.mindmoveStreamingFrequencyComboBox.setItemText(
            1, QCoreApplication.translate("MindMove", "16 Hz", None)
        )
        self.mindmoveStreamingFrequencyComboBox.setItemText(
            2, QCoreApplication.translate("MindMove", "8 Hz", None)
        )
        self.mindmoveStreamingFrequencyComboBox.setItemText(
            3, QCoreApplication.translate("MindMove", "4 Hz", None)
        )
        self.mindmoveStreamingFrequencyComboBox.setItemText(
            4, QCoreApplication.translate("MindMove", "2 Hz", None)
        )
        self.mindmoveStreamingFrequencyComboBox.setItemText(
            5, QCoreApplication.translate("MindMove", "1 Hz", None)
        )

        self.mindmoveUDPIPLineEdit.setText(
            QCoreApplication.translate("MindMove", "127.0.0.1", None)
        )
        self.label_18.setText(QCoreApplication.translate("MindMove", "IP", None))
        self.virtualHandInterfaceGroupBox.setTitle(
            QCoreApplication.translate("MindMove", "Virtual Hand Interface", None)
        )
        self.virtualHandInterfaceIPLineEdit.setText(
            QCoreApplication.translate("MindMove", "127.0.0.1", None)
        )
        self.label_3.setText(QCoreApplication.translate("MindMove", "IP", None))
        self.label_4.setText(QCoreApplication.translate("MindMove", "Port", None))
        self.virtualHandInterfacePortLineEdit.setText(
            QCoreApplication.translate("MindMove", "1236", None)
        )
        self.groupBox_2.setTitle(
            QCoreApplication.translate("MindMove", "Commands", None)
        )
        # if QT_CONFIG(tooltip)
        self.configureAppUdpSocketPushButton.setToolTip("")
        # endif // QT_CONFIG(tooltip)
        self.configureAppUdpSocketPushButton.setText(
            QCoreApplication.translate("MindMove", "Configure", None)
        )
        # if QT_CONFIG(tooltip)
        self.outputToggleStreamingPushButton.setToolTip("")
        # endif // QT_CONFIG(tooltip)
        self.outputToggleStreamingPushButton.setText(
            QCoreApplication.translate("MindMove", "Start Streaming", None)
        )
        self.mindMoveTabWidget.setTabText(
            self.mindMoveTabWidget.indexOf(self.virtualHandWidget),
            QCoreApplication.translate("MindMove", "Virtual Hand Interface", None),
        )
        self.reviewRecordingGroupBox.setTitle(
            QCoreApplication.translate("MindMove", "Review Recording", None)
        )
        self.label_5.setText(QCoreApplication.translate("MindMove", "Task", None))
        self.reviewRecordingTaskLabel.setText(
            QCoreApplication.translate("MindMove", "Placeholder", None)
        )
        self.label_2.setText(QCoreApplication.translate("MindMove", "Label", None))
        self.reviewRecordingAcceptPushButton.setText(
            QCoreApplication.translate("MindMove", "Accept", None)
        )
        self.reviewRecordingRejectPushButton.setText(
            QCoreApplication.translate("MindMove", "Reject", None)
        )
        self.recordRecordingGroupBox.setTitle(
            QCoreApplication.translate("MindMove", "Record", None)
        )
        self.label.setText(QCoreApplication.translate("MindMove", "Task", None))
        self.recordTaskComboBox.setItemText(
            0, QCoreApplication.translate("MindMove", "Rest", None)
        )
        self.recordTaskComboBox.setItemText(
            1, QCoreApplication.translate("MindMove", "Fist", None)
        )
        self.recordTaskComboBox.setItemText(
            2, QCoreApplication.translate("MindMove", "Pinch", None)
        )
        self.recordTaskComboBox.setItemText(
            3, QCoreApplication.translate("MindMove", "3FPinch", None)
        )
        self.recordTaskComboBox.setItemText(
            4, QCoreApplication.translate("MindMove", "Thumb", None)
        )
        self.recordTaskComboBox.setItemText(
            5, QCoreApplication.translate("MindMove", "Index", None)
        )
        self.recordTaskComboBox.setItemText(
            6, QCoreApplication.translate("MindMove", "Middle", None)
        )
        self.recordTaskComboBox.setItemText(
            7, QCoreApplication.translate("MindMove", "Ring", None)
        )
        self.recordTaskComboBox.setItemText(
            8, QCoreApplication.translate("MindMove", "Pinky", None)
        )

        self.recordRecordPushButton.setText(
            QCoreApplication.translate("MindMove", "Record", None)
        )
        self.label_7.setText(QCoreApplication.translate("MindMove", "Duration", None))
        self.trainingCreateDatasetGroupBox.setTitle(
            QCoreApplication.translate("MindMove", "Create Training Dataset", None)
        )
        self.trainingCreateDatasetsSelectRecordingsPushButton.setText(
            QCoreApplication.translate("MindMove", "Select Recordings", None)
        )
        self.trainingCreateDatasetPushButton.setText(
            QCoreApplication.translate("MindMove", "Create Dataset", None)
        )
        self.label_6.setText(QCoreApplication.translate("MindMove", "Label", None))
        self.trainingTrainModelGroupBox.setTitle(
            QCoreApplication.translate("MindMove", "Train Model", None)
        )
        self.trainingTrainNewModelRadioButton.setText(
            QCoreApplication.translate("MindMove", "New", None)
        )
        self.trainingTrainModelPushButton.setText(
            QCoreApplication.translate("MindMove", "Train", None)
        )
        self.trainingSelectDatasetPushButton.setText(
            QCoreApplication.translate("MindMove", "Select Dataset", None)
        )
        self.trainingSelectedDatasetLabel.setText(
            QCoreApplication.translate("MindMove", "Placeholder", None)
        )
        self.trainingLoadExistingModelPushButton.setText(
            QCoreApplication.translate("MindMove", "Load Model", None)
        )
        self.trainingLoadExistingModelLabel.setText(
            QCoreApplication.translate("MindMove", "Placeholder", None)
        )
        self.label_8.setText(QCoreApplication.translate("MindMove", "Label", None))
        self.trainingTrainExistingModelRadioButton.setText(
            QCoreApplication.translate("MindMove", "Existing", None)
        )
        self.onlineLoadModelGroupBox.setTitle(
            QCoreApplication.translate("MindMove", "Load Model", None)
        )
        self.onlineLoadModelPushButton.setText(
            QCoreApplication.translate("MindMove", "Select Model", None)
        )
        self.onlineModelLabel.setText(
            QCoreApplication.translate("MindMove", "Placeholder", None)
        )
        self.onlineCommandsGroupBox.setTitle(
            QCoreApplication.translate("MindMove", "Commands", None)
        )
        self.onlineRecordTogglePushButton.setText(
            QCoreApplication.translate("MindMove", "Start Recording", None)
        )
        self.protocolModeGroupBox.setTitle(
            QCoreApplication.translate("MindMove", "Mode", None)
        )
        self.protocolTrainingRadioButton.setText(
            QCoreApplication.translate("MindMove", "Training", None)
        )
        self.protocolOnlineRadioButton.setText(
            QCoreApplication.translate("MindMove", "Online", None)
        )
        self.protocolRecordRadioButton.setText(
            QCoreApplication.translate("MindMove", "Record", None)
        )
        self.mindMoveTabWidget.setTabText(
            self.mindMoveTabWidget.indexOf(self.procotolWidget),
            QCoreApplication.translate("MindMove", "Protocol", None),
        )

    # retranslateUi
