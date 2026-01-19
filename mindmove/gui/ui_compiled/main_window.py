# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'main_window.ui'
##
## Created by: Qt User Interface Compiler version 6.7.0
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QCheckBox, QComboBox,
    QGridLayout, QGroupBox, QLabel, QLineEdit,
    QListWidget, QListWidgetItem, QMainWindow, QMenuBar,
    QProgressBar, QPushButton, QRadioButton, QSizePolicy,
    QSpacerItem, QSpinBox, QStackedWidget, QStatusBar,
    QTabWidget, QWidget)

from mindmove.device_interfaces.gui.muovi_widget import MuoviWidget
from mindmove.gui_custom_elements.vispy_plot_widget import VispyPlotWidget

class Ui_MindMove(object):
    def setupUi(self, MindMove):
        if not MindMove.objectName():
            MindMove.setObjectName(u"MindMove")
        MindMove.resize(804, 607)
        self.centralwidget = QWidget(MindMove)
        self.centralwidget.setObjectName(u"centralwidget")
        self.gridLayout = QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName(u"gridLayout")
        self.vispyPlotWidget = VispyPlotWidget(self.centralwidget)
        self.vispyPlotWidget.setObjectName(u"vispyPlotWidget")
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.vispyPlotWidget.sizePolicy().hasHeightForWidth())
        self.vispyPlotWidget.setSizePolicy(sizePolicy)
        self.gridLayout_6 = QGridLayout(self.vispyPlotWidget)
        self.gridLayout_6.setObjectName(u"gridLayout_6")
        self.vispyPlotEnabledCheckBox = QCheckBox(self.vispyPlotWidget)
        self.vispyPlotEnabledCheckBox.setObjectName(u"vispyPlotEnabledCheckBox")

        self.gridLayout_6.addWidget(self.vispyPlotEnabledCheckBox, 0, 0, 1, 1)

        self.verticalSpacer_3 = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.gridLayout_6.addItem(self.verticalSpacer_3, 1, 0, 1, 1)

        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.gridLayout_6.addItem(self.horizontalSpacer, 0, 1, 1, 1)


        self.gridLayout.addWidget(self.vispyPlotWidget, 0, 1, 1, 1)

        self.mindMoveTabWidget = QTabWidget(self.centralwidget)
        self.mindMoveTabWidget.setObjectName(u"mindMoveTabWidget")
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.mindMoveTabWidget.sizePolicy().hasHeightForWidth())
        self.mindMoveTabWidget.setSizePolicy(sizePolicy1)
        self.muoviWidget = MuoviWidget()
        self.muoviWidget.setObjectName(u"muoviWidget")
        self.mindMoveTabWidget.addTab(self.muoviWidget, "")
        self.virtualHandWidget = QWidget()
        self.virtualHandWidget.setObjectName(u"virtualHandWidget")
        self.gridLayout_4 = QGridLayout(self.virtualHandWidget)
        self.gridLayout_4.setObjectName(u"gridLayout_4")
        self.verticalSpacer_4 = QSpacerItem(20, 99, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.gridLayout_4.addItem(self.verticalSpacer_4, 3, 1, 1, 1)

        self.mindmoveUDPSocketGroupBox = QGroupBox(self.virtualHandWidget)
        self.mindmoveUDPSocketGroupBox.setObjectName(u"mindmoveUDPSocketGroupBox")
        self.gridLayout_15 = QGridLayout(self.mindmoveUDPSocketGroupBox)
        self.gridLayout_15.setObjectName(u"gridLayout_15")
        self.label_42 = QLabel(self.mindmoveUDPSocketGroupBox)
        self.label_42.setObjectName(u"label_42")

        self.gridLayout_15.addWidget(self.label_42, 3, 0, 1, 1)

        self.mindmoveUDPPortLineEdit = QLineEdit(self.mindmoveUDPSocketGroupBox)
        self.mindmoveUDPPortLineEdit.setObjectName(u"mindmoveUDPPortLineEdit")
        sizePolicy2 = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.mindmoveUDPPortLineEdit.sizePolicy().hasHeightForWidth())
        self.mindmoveUDPPortLineEdit.setSizePolicy(sizePolicy2)

        self.gridLayout_15.addWidget(self.mindmoveUDPPortLineEdit, 1, 1, 1, 1)

        self.label_19 = QLabel(self.mindmoveUDPSocketGroupBox)
        self.label_19.setObjectName(u"label_19")

        self.gridLayout_15.addWidget(self.label_19, 1, 0, 1, 1)

        self.mindmoveStreamingFrequencyComboBox = QComboBox(self.mindmoveUDPSocketGroupBox)
        self.mindmoveStreamingFrequencyComboBox.addItem("")
        self.mindmoveStreamingFrequencyComboBox.addItem("")
        self.mindmoveStreamingFrequencyComboBox.addItem("")
        self.mindmoveStreamingFrequencyComboBox.addItem("")
        self.mindmoveStreamingFrequencyComboBox.addItem("")
        self.mindmoveStreamingFrequencyComboBox.addItem("")
        self.mindmoveStreamingFrequencyComboBox.setObjectName(u"mindmoveStreamingFrequencyComboBox")

        self.gridLayout_15.addWidget(self.mindmoveStreamingFrequencyComboBox, 3, 1, 1, 1)

        self.mindmoveUDPIPLineEdit = QLineEdit(self.mindmoveUDPSocketGroupBox)
        self.mindmoveUDPIPLineEdit.setObjectName(u"mindmoveUDPIPLineEdit")
        sizePolicy2.setHeightForWidth(self.mindmoveUDPIPLineEdit.sizePolicy().hasHeightForWidth())
        self.mindmoveUDPIPLineEdit.setSizePolicy(sizePolicy2)

        self.gridLayout_15.addWidget(self.mindmoveUDPIPLineEdit, 0, 1, 1, 1)

        self.label_18 = QLabel(self.mindmoveUDPSocketGroupBox)
        self.label_18.setObjectName(u"label_18")

        self.gridLayout_15.addWidget(self.label_18, 0, 0, 1, 1)


        self.gridLayout_4.addWidget(self.mindmoveUDPSocketGroupBox, 0, 0, 1, 2)

        self.virtualHandInterfaceGroupBox = QGroupBox(self.virtualHandWidget)
        self.virtualHandInterfaceGroupBox.setObjectName(u"virtualHandInterfaceGroupBox")
        self.gridLayout_7 = QGridLayout(self.virtualHandInterfaceGroupBox)
        self.gridLayout_7.setObjectName(u"gridLayout_7")
        self.virtualHandInterfaceIPLineEdit = QLineEdit(self.virtualHandInterfaceGroupBox)
        self.virtualHandInterfaceIPLineEdit.setObjectName(u"virtualHandInterfaceIPLineEdit")
        sizePolicy2.setHeightForWidth(self.virtualHandInterfaceIPLineEdit.sizePolicy().hasHeightForWidth())
        self.virtualHandInterfaceIPLineEdit.setSizePolicy(sizePolicy2)

        self.gridLayout_7.addWidget(self.virtualHandInterfaceIPLineEdit, 0, 1, 1, 1)

        self.label_3 = QLabel(self.virtualHandInterfaceGroupBox)
        self.label_3.setObjectName(u"label_3")

        self.gridLayout_7.addWidget(self.label_3, 0, 0, 1, 1)

        self.label_4 = QLabel(self.virtualHandInterfaceGroupBox)
        self.label_4.setObjectName(u"label_4")

        self.gridLayout_7.addWidget(self.label_4, 1, 0, 1, 1)

        self.virtualHandInterfacePortLineEdit = QLineEdit(self.virtualHandInterfaceGroupBox)
        self.virtualHandInterfacePortLineEdit.setObjectName(u"virtualHandInterfacePortLineEdit")
        sizePolicy2.setHeightForWidth(self.virtualHandInterfacePortLineEdit.sizePolicy().hasHeightForWidth())
        self.virtualHandInterfacePortLineEdit.setSizePolicy(sizePolicy2)

        self.gridLayout_7.addWidget(self.virtualHandInterfacePortLineEdit, 1, 1, 1, 1)


        self.gridLayout_4.addWidget(self.virtualHandInterfaceGroupBox, 1, 0, 1, 2)

        self.groupBox_2 = QGroupBox(self.virtualHandWidget)
        self.groupBox_2.setObjectName(u"groupBox_2")
        self.gridLayout_8 = QGridLayout(self.groupBox_2)
        self.gridLayout_8.setObjectName(u"gridLayout_8")
        self.configureAppUdpSocketPushButton = QPushButton(self.groupBox_2)
        self.configureAppUdpSocketPushButton.setObjectName(u"configureAppUdpSocketPushButton")
        self.configureAppUdpSocketPushButton.setCheckable(True)

        self.gridLayout_8.addWidget(self.configureAppUdpSocketPushButton, 0, 0, 1, 1)

        self.outputToggleStreamingPushButton = QPushButton(self.groupBox_2)
        self.outputToggleStreamingPushButton.setObjectName(u"outputToggleStreamingPushButton")
        self.outputToggleStreamingPushButton.setToolTipDuration(5)
        self.outputToggleStreamingPushButton.setCheckable(True)
        self.outputToggleStreamingPushButton.setChecked(False)

        self.gridLayout_8.addWidget(self.outputToggleStreamingPushButton, 0, 1, 1, 1)


        self.gridLayout_4.addWidget(self.groupBox_2, 2, 0, 1, 2)

        self.mindMoveTabWidget.addTab(self.virtualHandWidget, "")
        self.procotolWidget = QWidget()
        self.procotolWidget.setObjectName(u"procotolWidget")
        self.gridLayout_2 = QGridLayout(self.procotolWidget)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.protocolModeStackedWidget = QStackedWidget(self.procotolWidget)
        self.protocolModeStackedWidget.setObjectName(u"protocolModeStackedWidget")
        self.recordWidget = QWidget()
        self.recordWidget.setObjectName(u"recordWidget")
        self.gridLayout_5 = QGridLayout(self.recordWidget)
        self.gridLayout_5.setObjectName(u"gridLayout_5")
        self.recordReviewRecordingStackedWidget = QStackedWidget(self.recordWidget)
        self.recordReviewRecordingStackedWidget.setObjectName(u"recordReviewRecordingStackedWidget")
        self.emptyWidget_2 = QWidget()
        self.emptyWidget_2.setObjectName(u"emptyWidget_2")
        self.recordReviewRecordingStackedWidget.addWidget(self.emptyWidget_2)
        self.reviewRecordingWidget = QWidget()
        self.reviewRecordingWidget.setObjectName(u"reviewRecordingWidget")
        self.gridLayout_11 = QGridLayout(self.reviewRecordingWidget)
        self.gridLayout_11.setObjectName(u"gridLayout_11")
        self.reviewRecordingGroupBox = QGroupBox(self.reviewRecordingWidget)
        self.reviewRecordingGroupBox.setObjectName(u"reviewRecordingGroupBox")
        sizePolicy1.setHeightForWidth(self.reviewRecordingGroupBox.sizePolicy().hasHeightForWidth())
        self.reviewRecordingGroupBox.setSizePolicy(sizePolicy1)
        self.gridLayout_10 = QGridLayout(self.reviewRecordingGroupBox)
        self.gridLayout_10.setObjectName(u"gridLayout_10")
        self.label_5 = QLabel(self.reviewRecordingGroupBox)
        self.label_5.setObjectName(u"label_5")

        self.gridLayout_10.addWidget(self.label_5, 0, 0, 1, 1)

        self.reviewRecordingTaskLabel = QLabel(self.reviewRecordingGroupBox)
        self.reviewRecordingTaskLabel.setObjectName(u"reviewRecordingTaskLabel")

        self.gridLayout_10.addWidget(self.reviewRecordingTaskLabel, 0, 1, 1, 1)

        self.label_2 = QLabel(self.reviewRecordingGroupBox)
        self.label_2.setObjectName(u"label_2")

        self.gridLayout_10.addWidget(self.label_2, 1, 0, 1, 1)

        self.reviewRecordingKinematicsPlotWidget = VispyPlotWidget(self.reviewRecordingGroupBox)
        self.reviewRecordingKinematicsPlotWidget.setObjectName(u"reviewRecordingKinematicsPlotWidget")
        sizePolicy1.setHeightForWidth(self.reviewRecordingKinematicsPlotWidget.sizePolicy().hasHeightForWidth())
        self.reviewRecordingKinematicsPlotWidget.setSizePolicy(sizePolicy1)

        self.gridLayout_10.addWidget(self.reviewRecordingKinematicsPlotWidget, 4, 0, 1, 2)

        self.reviewRecordingLabelLineEdit = QLineEdit(self.reviewRecordingGroupBox)
        self.reviewRecordingLabelLineEdit.setObjectName(u"reviewRecordingLabelLineEdit")
        sizePolicy3 = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        sizePolicy3.setHorizontalStretch(0)
        sizePolicy3.setVerticalStretch(0)
        sizePolicy3.setHeightForWidth(self.reviewRecordingLabelLineEdit.sizePolicy().hasHeightForWidth())
        self.reviewRecordingLabelLineEdit.setSizePolicy(sizePolicy3)

        self.gridLayout_10.addWidget(self.reviewRecordingLabelLineEdit, 1, 1, 1, 1)

        self.reviewRecordingAcceptPushButton = QPushButton(self.reviewRecordingGroupBox)
        self.reviewRecordingAcceptPushButton.setObjectName(u"reviewRecordingAcceptPushButton")
        sizePolicy4 = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        sizePolicy4.setHorizontalStretch(0)
        sizePolicy4.setVerticalStretch(0)
        sizePolicy4.setHeightForWidth(self.reviewRecordingAcceptPushButton.sizePolicy().hasHeightForWidth())
        self.reviewRecordingAcceptPushButton.setSizePolicy(sizePolicy4)
        self.reviewRecordingAcceptPushButton.setStyleSheet(u"color: rgb(0, 0, 0); background-color: rgb(170, 255, 0);")

        self.gridLayout_10.addWidget(self.reviewRecordingAcceptPushButton, 5, 0, 1, 1)

        self.reviewRecordingEMGPlotWidget = VispyPlotWidget(self.reviewRecordingGroupBox)
        self.reviewRecordingEMGPlotWidget.setObjectName(u"reviewRecordingEMGPlotWidget")
        sizePolicy1.setHeightForWidth(self.reviewRecordingEMGPlotWidget.sizePolicy().hasHeightForWidth())
        self.reviewRecordingEMGPlotWidget.setSizePolicy(sizePolicy1)

        self.gridLayout_10.addWidget(self.reviewRecordingEMGPlotWidget, 3, 0, 1, 2)

        self.reviewRecordingRejectPushButton = QPushButton(self.reviewRecordingGroupBox)
        self.reviewRecordingRejectPushButton.setObjectName(u"reviewRecordingRejectPushButton")
        sizePolicy4.setHeightForWidth(self.reviewRecordingRejectPushButton.sizePolicy().hasHeightForWidth())
        self.reviewRecordingRejectPushButton.setSizePolicy(sizePolicy4)
        self.reviewRecordingRejectPushButton.setLayoutDirection(Qt.LayoutDirection.LeftToRight)
        self.reviewRecordingRejectPushButton.setStyleSheet(u"background-color: rgb(255, 0, 0);\n"
"color: rgb(0, 0, 0);")

        self.gridLayout_10.addWidget(self.reviewRecordingRejectPushButton, 5, 1, 1, 1)


        self.gridLayout_11.addWidget(self.reviewRecordingGroupBox, 0, 0, 1, 1)

        self.recordReviewRecordingStackedWidget.addWidget(self.reviewRecordingWidget)

        self.gridLayout_5.addWidget(self.recordReviewRecordingStackedWidget, 1, 0, 1, 1)

        self.recordRecordingGroupBox = QGroupBox(self.recordWidget)
        self.recordRecordingGroupBox.setObjectName(u"recordRecordingGroupBox")
        self.gridLayout_9 = QGridLayout(self.recordRecordingGroupBox)
        self.gridLayout_9.setObjectName(u"gridLayout_9")
        self.label = QLabel(self.recordRecordingGroupBox)
        self.label.setObjectName(u"label")

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
        self.recordTaskComboBox.setObjectName(u"recordTaskComboBox")

        self.gridLayout_9.addWidget(self.recordTaskComboBox, 0, 1, 1, 1)

        self.recordRecordPushButton = QPushButton(self.recordRecordingGroupBox)
        self.recordRecordPushButton.setObjectName(u"recordRecordPushButton")
        self.recordRecordPushButton.setCheckable(True)

        self.gridLayout_9.addWidget(self.recordRecordPushButton, 2, 0, 1, 2)

        self.recordEMGProgressBar = QProgressBar(self.recordRecordingGroupBox)
        self.recordEMGProgressBar.setObjectName(u"recordEMGProgressBar")
        self.recordEMGProgressBar.setValue(24)

        self.gridLayout_9.addWidget(self.recordEMGProgressBar, 3, 0, 1, 1)

        self.recordKinematicsProgressBar = QProgressBar(self.recordRecordingGroupBox)
        self.recordKinematicsProgressBar.setObjectName(u"recordKinematicsProgressBar")
        self.recordKinematicsProgressBar.setValue(24)

        self.gridLayout_9.addWidget(self.recordKinematicsProgressBar, 3, 1, 1, 1)

        self.label_7 = QLabel(self.recordRecordingGroupBox)
        self.label_7.setObjectName(u"label_7")

        self.gridLayout_9.addWidget(self.label_7, 1, 0, 1, 1)

        self.recordDurationSpinBox = QSpinBox(self.recordRecordingGroupBox)
        self.recordDurationSpinBox.setObjectName(u"recordDurationSpinBox")
        self.recordDurationSpinBox.setValue(10)

        self.gridLayout_9.addWidget(self.recordDurationSpinBox, 1, 1, 1, 1)


        self.gridLayout_5.addWidget(self.recordRecordingGroupBox, 0, 0, 1, 2)

        self.protocolModeStackedWidget.addWidget(self.recordWidget)
        self.trainingWidget = QWidget()
        self.trainingWidget.setObjectName(u"trainingWidget")
        self.gridLayout_12 = QGridLayout(self.trainingWidget)
        self.gridLayout_12.setObjectName(u"gridLayout_12")
        self.trainingTemplateExtractionGroupBox = QGroupBox(self.trainingWidget)
        self.trainingTemplateExtractionGroupBox.setObjectName(u"trainingTemplateExtractionGroupBox")
        self.gridLayout_20 = QGridLayout(self.trainingTemplateExtractionGroupBox)
        self.gridLayout_20.setObjectName(u"gridLayout_20")
        self.label_9 = QLabel(self.trainingTemplateExtractionGroupBox)
        self.label_9.setObjectName(u"label_9")

        self.gridLayout_20.addWidget(self.label_9, 0, 0, 1, 1)

        self.trainingTemplateClassComboBox = QComboBox(self.trainingTemplateExtractionGroupBox)
        self.trainingTemplateClassComboBox.addItem("")
        self.trainingTemplateClassComboBox.addItem("")
        self.trainingTemplateClassComboBox.setObjectName(u"trainingTemplateClassComboBox")

        self.gridLayout_20.addWidget(self.trainingTemplateClassComboBox, 0, 1, 1, 1)

        self.label_12 = QLabel(self.trainingTemplateExtractionGroupBox)
        self.label_12.setObjectName(u"label_12")

        self.gridLayout_20.addWidget(self.label_12, 1, 0, 1, 1)

        self.trainingDataFormatComboBox = QComboBox(self.trainingTemplateExtractionGroupBox)
        self.trainingDataFormatComboBox.addItem("")
        self.trainingDataFormatComboBox.addItem("")
        self.trainingDataFormatComboBox.setObjectName(u"trainingDataFormatComboBox")

        self.gridLayout_20.addWidget(self.trainingDataFormatComboBox, 1, 1, 1, 1)

        self.label_10 = QLabel(self.trainingTemplateExtractionGroupBox)
        self.label_10.setObjectName(u"label_10")

        self.gridLayout_20.addWidget(self.label_10, 2, 0, 1, 1)

        self.trainingTemplateTypeComboBox = QComboBox(self.trainingTemplateExtractionGroupBox)
        self.trainingTemplateTypeComboBox.addItem("")
        self.trainingTemplateTypeComboBox.addItem("")
        self.trainingTemplateTypeComboBox.setObjectName(u"trainingTemplateTypeComboBox")

        self.gridLayout_20.addWidget(self.trainingTemplateTypeComboBox, 2, 1, 1, 1)

        self.trainingSelectRecordingsForExtractionPushButton = QPushButton(self.trainingTemplateExtractionGroupBox)
        self.trainingSelectRecordingsForExtractionPushButton.setObjectName(u"trainingSelectRecordingsForExtractionPushButton")

        self.gridLayout_20.addWidget(self.trainingSelectRecordingsForExtractionPushButton, 3, 0, 1, 1)

        self.trainingSelectedRecordingsForExtractionLabel = QLabel(self.trainingTemplateExtractionGroupBox)
        self.trainingSelectedRecordingsForExtractionLabel.setObjectName(u"trainingSelectedRecordingsForExtractionLabel")

        self.gridLayout_20.addWidget(self.trainingSelectedRecordingsForExtractionLabel, 3, 1, 1, 1)

        self.trainingExtractActivationsPushButton = QPushButton(self.trainingTemplateExtractionGroupBox)
        self.trainingExtractActivationsPushButton.setObjectName(u"trainingExtractActivationsPushButton")

        self.gridLayout_20.addWidget(self.trainingExtractActivationsPushButton, 4, 0, 1, 1)

        self.trainingActivationCountLabel = QLabel(self.trainingTemplateExtractionGroupBox)
        self.trainingActivationCountLabel.setObjectName(u"trainingActivationCountLabel")

        self.gridLayout_20.addWidget(self.trainingActivationCountLabel, 4, 1, 1, 1)

        self.label_11 = QLabel(self.trainingTemplateExtractionGroupBox)
        self.label_11.setObjectName(u"label_11")

        self.gridLayout_20.addWidget(self.label_11, 5, 0, 1, 1)

        self.trainingSelectionModeComboBox = QComboBox(self.trainingTemplateExtractionGroupBox)
        self.trainingSelectionModeComboBox.addItem("")
        self.trainingSelectionModeComboBox.addItem("")
        self.trainingSelectionModeComboBox.addItem("")
        self.trainingSelectionModeComboBox.setObjectName(u"trainingSelectionModeComboBox")

        self.gridLayout_20.addWidget(self.trainingSelectionModeComboBox, 5, 1, 1, 1)

        self.trainingActivationListWidget = QListWidget(self.trainingTemplateExtractionGroupBox)
        self.trainingActivationListWidget.setObjectName(u"trainingActivationListWidget")
        sizePolicy1.setHeightForWidth(self.trainingActivationListWidget.sizePolicy().hasHeightForWidth())
        self.trainingActivationListWidget.setSizePolicy(sizePolicy1)
        self.trainingActivationListWidget.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)

        self.gridLayout_20.addWidget(self.trainingActivationListWidget, 6, 0, 1, 2)

        self.trainingPlotSelectedPushButton = QPushButton(self.trainingTemplateExtractionGroupBox)
        self.trainingPlotSelectedPushButton.setObjectName(u"trainingPlotSelectedPushButton")

        self.gridLayout_20.addWidget(self.trainingPlotSelectedPushButton, 7, 0, 1, 1)

        self.trainingSelectTemplatesPushButton = QPushButton(self.trainingTemplateExtractionGroupBox)
        self.trainingSelectTemplatesPushButton.setObjectName(u"trainingSelectTemplatesPushButton")

        self.gridLayout_20.addWidget(self.trainingSelectTemplatesPushButton, 7, 1, 1, 1)

        self.trainingTemplateCountLabel = QLabel(self.trainingTemplateExtractionGroupBox)
        self.trainingTemplateCountLabel.setObjectName(u"trainingTemplateCountLabel")

        self.gridLayout_20.addWidget(self.trainingTemplateCountLabel, 8, 0, 1, 1)

        self.trainingSaveTemplatesPushButton = QPushButton(self.trainingTemplateExtractionGroupBox)
        self.trainingSaveTemplatesPushButton.setObjectName(u"trainingSaveTemplatesPushButton")

        self.gridLayout_20.addWidget(self.trainingSaveTemplatesPushButton, 9, 0, 1, 1)

        self.trainingClearExtractionPushButton = QPushButton(self.trainingTemplateExtractionGroupBox)
        self.trainingClearExtractionPushButton.setObjectName(u"trainingClearExtractionPushButton")

        self.gridLayout_20.addWidget(self.trainingClearExtractionPushButton, 9, 1, 1, 1)

        self.trainingExtractionProgressBar = QProgressBar(self.trainingTemplateExtractionGroupBox)
        self.trainingExtractionProgressBar.setObjectName(u"trainingExtractionProgressBar")
        self.trainingExtractionProgressBar.setValue(0)

        self.gridLayout_20.addWidget(self.trainingExtractionProgressBar, 10, 0, 1, 2)


        self.gridLayout_12.addWidget(self.trainingTemplateExtractionGroupBox, 0, 0, 1, 1)

        self.trainingCreateDatasetGroupBox = QGroupBox(self.trainingWidget)
        self.trainingCreateDatasetGroupBox.setObjectName(u"trainingCreateDatasetGroupBox")
        self.gridLayout_13 = QGridLayout(self.trainingCreateDatasetGroupBox)
        self.gridLayout_13.setObjectName(u"gridLayout_13")
        self.trainingCreateDatasetsSelectRecordingsPushButton = QPushButton(self.trainingCreateDatasetGroupBox)
        self.trainingCreateDatasetsSelectRecordingsPushButton.setObjectName(u"trainingCreateDatasetsSelectRecordingsPushButton")

        self.gridLayout_13.addWidget(self.trainingCreateDatasetsSelectRecordingsPushButton, 0, 0, 1, 1)

        self.trainingCreateDatasetPushButton = QPushButton(self.trainingCreateDatasetGroupBox)
        self.trainingCreateDatasetPushButton.setObjectName(u"trainingCreateDatasetPushButton")

        self.gridLayout_13.addWidget(self.trainingCreateDatasetPushButton, 3, 0, 1, 1)

        self.label_6 = QLabel(self.trainingCreateDatasetGroupBox)
        self.label_6.setObjectName(u"label_6")

        self.gridLayout_13.addWidget(self.label_6, 2, 0, 1, 1)

        self.trainingCreateDatasetSelectedRecordingsListWidget = QListWidget(self.trainingCreateDatasetGroupBox)
        self.trainingCreateDatasetSelectedRecordingsListWidget.setObjectName(u"trainingCreateDatasetSelectedRecordingsListWidget")
        self.trainingCreateDatasetSelectedRecordingsListWidget.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)

        self.gridLayout_13.addWidget(self.trainingCreateDatasetSelectedRecordingsListWidget, 1, 0, 1, 2)

        self.trainingCreateDatasetLabelLineEdit = QLineEdit(self.trainingCreateDatasetGroupBox)
        self.trainingCreateDatasetLabelLineEdit.setObjectName(u"trainingCreateDatasetLabelLineEdit")

        self.gridLayout_13.addWidget(self.trainingCreateDatasetLabelLineEdit, 2, 1, 1, 1)

        self.trainingCreateDatasetProgressBar = QProgressBar(self.trainingCreateDatasetGroupBox)
        self.trainingCreateDatasetProgressBar.setObjectName(u"trainingCreateDatasetProgressBar")
        self.trainingCreateDatasetProgressBar.setValue(24)

        self.gridLayout_13.addWidget(self.trainingCreateDatasetProgressBar, 3, 1, 1, 1)


        self.gridLayout_12.addWidget(self.trainingCreateDatasetGroupBox, 1, 0, 1, 1)

        self.trainingTrainModelGroupBox = QGroupBox(self.trainingWidget)
        self.trainingTrainModelGroupBox.setObjectName(u"trainingTrainModelGroupBox")
        self.gridLayout_14 = QGridLayout(self.trainingTrainModelGroupBox)
        self.gridLayout_14.setObjectName(u"gridLayout_14")
        self.trainingTrainNewModelRadioButton = QRadioButton(self.trainingTrainModelGroupBox)
        self.trainingTrainNewModelRadioButton.setObjectName(u"trainingTrainNewModelRadioButton")
        self.trainingTrainNewModelRadioButton.setChecked(True)

        self.gridLayout_14.addWidget(self.trainingTrainNewModelRadioButton, 0, 1, 1, 1)

        self.trainingTrainModelPushButton = QPushButton(self.trainingTrainModelGroupBox)
        self.trainingTrainModelPushButton.setObjectName(u"trainingTrainModelPushButton")

        self.gridLayout_14.addWidget(self.trainingTrainModelPushButton, 4, 1, 1, 1)

        self.trainingSelectDatasetPushButton = QPushButton(self.trainingTrainModelGroupBox)
        self.trainingSelectDatasetPushButton.setObjectName(u"trainingSelectDatasetPushButton")

        self.gridLayout_14.addWidget(self.trainingSelectDatasetPushButton, 2, 1, 1, 1)

        self.trainingProgressBar = QProgressBar(self.trainingTrainModelGroupBox)
        self.trainingProgressBar.setObjectName(u"trainingProgressBar")
        self.trainingProgressBar.setValue(24)

        self.gridLayout_14.addWidget(self.trainingProgressBar, 4, 2, 1, 1)

        self.trainingSelectedDatasetLabel = QLabel(self.trainingTrainModelGroupBox)
        self.trainingSelectedDatasetLabel.setObjectName(u"trainingSelectedDatasetLabel")

        self.gridLayout_14.addWidget(self.trainingSelectedDatasetLabel, 2, 2, 1, 1)

        self.trainingModelLabelLineEdit = QLineEdit(self.trainingTrainModelGroupBox)
        self.trainingModelLabelLineEdit.setObjectName(u"trainingModelLabelLineEdit")

        self.gridLayout_14.addWidget(self.trainingModelLabelLineEdit, 3, 2, 1, 1)

        self.trainingTrainModelStackedWidget = QStackedWidget(self.trainingTrainModelGroupBox)
        self.trainingTrainModelStackedWidget.setObjectName(u"trainingTrainModelStackedWidget")
        sizePolicy2.setHeightForWidth(self.trainingTrainModelStackedWidget.sizePolicy().hasHeightForWidth())
        self.trainingTrainModelStackedWidget.setSizePolicy(sizePolicy2)
        self.emptyWidget = QWidget()
        self.emptyWidget.setObjectName(u"emptyWidget")
        self.trainingTrainModelStackedWidget.addWidget(self.emptyWidget)
        self.trainingLoadExistingModelWidget = QWidget()
        self.trainingLoadExistingModelWidget.setObjectName(u"trainingLoadExistingModelWidget")
        self.gridLayout_17 = QGridLayout(self.trainingLoadExistingModelWidget)
        self.gridLayout_17.setObjectName(u"gridLayout_17")
        self.trainingLoadExistingModelPushButton = QPushButton(self.trainingLoadExistingModelWidget)
        self.trainingLoadExistingModelPushButton.setObjectName(u"trainingLoadExistingModelPushButton")

        self.gridLayout_17.addWidget(self.trainingLoadExistingModelPushButton, 0, 0, 1, 1)

        self.trainingLoadExistingModelLabel = QLabel(self.trainingLoadExistingModelWidget)
        self.trainingLoadExistingModelLabel.setObjectName(u"trainingLoadExistingModelLabel")

        self.gridLayout_17.addWidget(self.trainingLoadExistingModelLabel, 0, 1, 1, 1)

        self.trainingTrainModelStackedWidget.addWidget(self.trainingLoadExistingModelWidget)

        self.gridLayout_14.addWidget(self.trainingTrainModelStackedWidget, 1, 1, 1, 2)

        self.label_8 = QLabel(self.trainingTrainModelGroupBox)
        self.label_8.setObjectName(u"label_8")

        self.gridLayout_14.addWidget(self.label_8, 3, 1, 1, 1)

        self.trainingTrainExistingModelRadioButton = QRadioButton(self.trainingTrainModelGroupBox)
        self.trainingTrainExistingModelRadioButton.setObjectName(u"trainingTrainExistingModelRadioButton")

        self.gridLayout_14.addWidget(self.trainingTrainExistingModelRadioButton, 0, 2, 1, 1)


        self.gridLayout_12.addWidget(self.trainingTrainModelGroupBox, 2, 0, 1, 1)

        self.protocolModeStackedWidget.addWidget(self.trainingWidget)
        self.onlineWidget = QWidget()
        self.onlineWidget.setObjectName(u"onlineWidget")
        self.gridLayout_16 = QGridLayout(self.onlineWidget)
        self.gridLayout_16.setObjectName(u"gridLayout_16")
        self.onlineLoadModelGroupBox = QGroupBox(self.onlineWidget)
        self.onlineLoadModelGroupBox.setObjectName(u"onlineLoadModelGroupBox")
        self.gridLayout_18 = QGridLayout(self.onlineLoadModelGroupBox)
        self.gridLayout_18.setObjectName(u"gridLayout_18")
        self.onlineLoadModelPushButton = QPushButton(self.onlineLoadModelGroupBox)
        self.onlineLoadModelPushButton.setObjectName(u"onlineLoadModelPushButton")
        self.onlineLoadModelPushButton.setCheckable(False)

        self.gridLayout_18.addWidget(self.onlineLoadModelPushButton, 0, 0, 1, 1)

        self.onlineModelLabel = QLabel(self.onlineLoadModelGroupBox)
        self.onlineModelLabel.setObjectName(u"onlineModelLabel")

        self.gridLayout_18.addWidget(self.onlineModelLabel, 0, 1, 1, 1)


        self.gridLayout_16.addWidget(self.onlineLoadModelGroupBox, 0, 0, 1, 2)

        self.onlineCommandsGroupBox = QGroupBox(self.onlineWidget)
        self.onlineCommandsGroupBox.setObjectName(u"onlineCommandsGroupBox")
        self.gridLayout_19 = QGridLayout(self.onlineCommandsGroupBox)
        self.gridLayout_19.setObjectName(u"gridLayout_19")
        self.onlineRecordTogglePushButton = QPushButton(self.onlineCommandsGroupBox)
        self.onlineRecordTogglePushButton.setObjectName(u"onlineRecordTogglePushButton")
        self.onlineRecordTogglePushButton.setCheckable(True)

        self.gridLayout_19.addWidget(self.onlineRecordTogglePushButton, 0, 0, 1, 1)


        self.gridLayout_16.addWidget(self.onlineCommandsGroupBox, 1, 0, 1, 2)

        self.verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.gridLayout_16.addItem(self.verticalSpacer, 2, 0, 1, 1)

        self.protocolModeStackedWidget.addWidget(self.onlineWidget)

        self.gridLayout_2.addWidget(self.protocolModeStackedWidget, 1, 0, 1, 1)

        self.protocolModeGroupBox = QGroupBox(self.procotolWidget)
        self.protocolModeGroupBox.setObjectName(u"protocolModeGroupBox")
        self.gridLayout_3 = QGridLayout(self.protocolModeGroupBox)
        self.gridLayout_3.setObjectName(u"gridLayout_3")
        self.protocolTrainingRadioButton = QRadioButton(self.protocolModeGroupBox)
        self.protocolTrainingRadioButton.setObjectName(u"protocolTrainingRadioButton")

        self.gridLayout_3.addWidget(self.protocolTrainingRadioButton, 0, 1, 1, 1)

        self.protocolOnlineRadioButton = QRadioButton(self.protocolModeGroupBox)
        self.protocolOnlineRadioButton.setObjectName(u"protocolOnlineRadioButton")

        self.gridLayout_3.addWidget(self.protocolOnlineRadioButton, 0, 2, 1, 1)

        self.protocolRecordRadioButton = QRadioButton(self.protocolModeGroupBox)
        self.protocolRecordRadioButton.setObjectName(u"protocolRecordRadioButton")
        self.protocolRecordRadioButton.setChecked(True)

        self.gridLayout_3.addWidget(self.protocolRecordRadioButton, 0, 0, 1, 1)


        self.gridLayout_2.addWidget(self.protocolModeGroupBox, 0, 0, 1, 1)

        self.verticalSpacer_2 = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.gridLayout_2.addItem(self.verticalSpacer_2, 2, 0, 1, 1)

        self.mindMoveTabWidget.addTab(self.procotolWidget, "")

        self.gridLayout.addWidget(self.mindMoveTabWidget, 0, 0, 1, 1)

        MindMove.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MindMove)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 804, 33))
        MindMove.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MindMove)
        self.statusbar.setObjectName(u"statusbar")
        MindMove.setStatusBar(self.statusbar)

        self.retranslateUi(MindMove)

        self.mindMoveTabWidget.setCurrentIndex(1)
        self.protocolModeStackedWidget.setCurrentIndex(2)
        self.recordReviewRecordingStackedWidget.setCurrentIndex(1)
        self.trainingTrainModelStackedWidget.setCurrentIndex(1)


        QMetaObject.connectSlotsByName(MindMove)
    # setupUi

    def retranslateUi(self, MindMove):
        MindMove.setWindowTitle(QCoreApplication.translate("MindMove", u"MindMove", None))
        self.vispyPlotEnabledCheckBox.setText(QCoreApplication.translate("MindMove", u"Plot Enabled", None))
        self.mindMoveTabWidget.setTabText(self.mindMoveTabWidget.indexOf(self.muoviWidget), QCoreApplication.translate("MindMove", u"Devices", None))
        self.mindmoveUDPSocketGroupBox.setTitle(QCoreApplication.translate("MindMove", u"MindMove UDP Socket Settings", None))
        self.label_42.setText(QCoreApplication.translate("MindMove", u"Streaming Frequency", None))
        self.mindmoveUDPPortLineEdit.setText(QCoreApplication.translate("MindMove", u"1233", None))
        self.label_19.setText(QCoreApplication.translate("MindMove", u"Port", None))
        self.mindmoveStreamingFrequencyComboBox.setItemText(0, QCoreApplication.translate("MindMove", u"32 Hz", None))
        self.mindmoveStreamingFrequencyComboBox.setItemText(1, QCoreApplication.translate("MindMove", u"16 Hz", None))
        self.mindmoveStreamingFrequencyComboBox.setItemText(2, QCoreApplication.translate("MindMove", u"8 Hz", None))
        self.mindmoveStreamingFrequencyComboBox.setItemText(3, QCoreApplication.translate("MindMove", u"4 Hz", None))
        self.mindmoveStreamingFrequencyComboBox.setItemText(4, QCoreApplication.translate("MindMove", u"2 Hz", None))
        self.mindmoveStreamingFrequencyComboBox.setItemText(5, QCoreApplication.translate("MindMove", u"1 Hz", None))

        self.mindmoveUDPIPLineEdit.setText(QCoreApplication.translate("MindMove", u"127.0.0.1", None))
        self.label_18.setText(QCoreApplication.translate("MindMove", u"IP", None))
        self.virtualHandInterfaceGroupBox.setTitle(QCoreApplication.translate("MindMove", u"Virtual Hand Interface", None))
        self.virtualHandInterfaceIPLineEdit.setText(QCoreApplication.translate("MindMove", u"127.0.0.1", None))
        self.label_3.setText(QCoreApplication.translate("MindMove", u"IP", None))
        self.label_4.setText(QCoreApplication.translate("MindMove", u"Port", None))
        self.virtualHandInterfacePortLineEdit.setText(QCoreApplication.translate("MindMove", u"1236", None))
        self.groupBox_2.setTitle(QCoreApplication.translate("MindMove", u"Commands", None))
#if QT_CONFIG(tooltip)
        self.configureAppUdpSocketPushButton.setToolTip("")
#endif // QT_CONFIG(tooltip)
        self.configureAppUdpSocketPushButton.setText(QCoreApplication.translate("MindMove", u"Configure", None))
#if QT_CONFIG(tooltip)
        self.outputToggleStreamingPushButton.setToolTip("")
#endif // QT_CONFIG(tooltip)
        self.outputToggleStreamingPushButton.setText(QCoreApplication.translate("MindMove", u"Start Streaming", None))
        self.mindMoveTabWidget.setTabText(self.mindMoveTabWidget.indexOf(self.virtualHandWidget), QCoreApplication.translate("MindMove", u"Virtual Hand Interface", None))
        self.reviewRecordingGroupBox.setTitle(QCoreApplication.translate("MindMove", u"Review Recording", None))
        self.label_5.setText(QCoreApplication.translate("MindMove", u"Task", None))
        self.reviewRecordingTaskLabel.setText(QCoreApplication.translate("MindMove", u"Placeholder", None))
        self.label_2.setText(QCoreApplication.translate("MindMove", u"Label", None))
        self.reviewRecordingAcceptPushButton.setText(QCoreApplication.translate("MindMove", u"Accept", None))
        self.reviewRecordingRejectPushButton.setText(QCoreApplication.translate("MindMove", u"Reject", None))
        self.recordRecordingGroupBox.setTitle(QCoreApplication.translate("MindMove", u"Record", None))
        self.label.setText(QCoreApplication.translate("MindMove", u"Task", None))
        self.recordTaskComboBox.setItemText(0, QCoreApplication.translate("MindMove", u"Rest", None))
        self.recordTaskComboBox.setItemText(1, QCoreApplication.translate("MindMove", u"Fist", None))
        self.recordTaskComboBox.setItemText(2, QCoreApplication.translate("MindMove", u"Pinch", None))
        self.recordTaskComboBox.setItemText(3, QCoreApplication.translate("MindMove", u"3FPinch", None))
        self.recordTaskComboBox.setItemText(4, QCoreApplication.translate("MindMove", u"Thumb", None))
        self.recordTaskComboBox.setItemText(5, QCoreApplication.translate("MindMove", u"Index", None))
        self.recordTaskComboBox.setItemText(6, QCoreApplication.translate("MindMove", u"Middle", None))
        self.recordTaskComboBox.setItemText(7, QCoreApplication.translate("MindMove", u"Ring", None))
        self.recordTaskComboBox.setItemText(8, QCoreApplication.translate("MindMove", u"Pinky", None))

        self.recordRecordPushButton.setText(QCoreApplication.translate("MindMove", u"Record", None))
        self.label_7.setText(QCoreApplication.translate("MindMove", u"Duration", None))
        self.trainingTemplateExtractionGroupBox.setTitle(QCoreApplication.translate("MindMove", u"Template Extraction", None))
        self.label_9.setText(QCoreApplication.translate("MindMove", u"Class", None))
        self.trainingTemplateClassComboBox.setItemText(0, QCoreApplication.translate("MindMove", u"Open", None))
        self.trainingTemplateClassComboBox.setItemText(1, QCoreApplication.translate("MindMove", u"Closed", None))

        self.label_12.setText(QCoreApplication.translate("MindMove", u"Data Format", None))
        self.trainingDataFormatComboBox.setItemText(0, QCoreApplication.translate("MindMove", u"Auto-detect (single files)", None))
        self.trainingDataFormatComboBox.setItemText(1, QCoreApplication.translate("MindMove", u"Legacy (EMG + GT folders)", None))

        self.label_10.setText(QCoreApplication.translate("MindMove", u"Template Type", None))
        self.trainingTemplateTypeComboBox.setItemText(0, QCoreApplication.translate("MindMove", u"Hold Only (skip 0.5s)", None))
        self.trainingTemplateTypeComboBox.setItemText(1, QCoreApplication.translate("MindMove", u"Onset + Hold (start -0.2s)", None))

        self.trainingSelectRecordingsForExtractionPushButton.setText(QCoreApplication.translate("MindMove", u"Select Recording(s)", None))
        self.trainingSelectedRecordingsForExtractionLabel.setText(QCoreApplication.translate("MindMove", u"No recordings selected", None))
        self.trainingExtractActivationsPushButton.setText(QCoreApplication.translate("MindMove", u"Extract Activations", None))
        self.trainingActivationCountLabel.setText(QCoreApplication.translate("MindMove", u"0 activations found", None))
        self.label_11.setText(QCoreApplication.translate("MindMove", u"Selection Mode", None))
        self.trainingSelectionModeComboBox.setItemText(0, QCoreApplication.translate("MindMove", u"Manual Review", None))
        self.trainingSelectionModeComboBox.setItemText(1, QCoreApplication.translate("MindMove", u"Auto (longest)", None))
        self.trainingSelectionModeComboBox.setItemText(2, QCoreApplication.translate("MindMove", u"First 20", None))

        self.trainingPlotSelectedPushButton.setText(QCoreApplication.translate("MindMove", u"Plot Selected", None))
        self.trainingSelectTemplatesPushButton.setText(QCoreApplication.translate("MindMove", u"Select as Templates", None))
        self.trainingTemplateCountLabel.setText(QCoreApplication.translate("MindMove", u"0/20 templates", None))
        self.trainingSaveTemplatesPushButton.setText(QCoreApplication.translate("MindMove", u"Save Templates", None))
        self.trainingClearExtractionPushButton.setText(QCoreApplication.translate("MindMove", u"Clear", None))
        self.trainingCreateDatasetGroupBox.setTitle(QCoreApplication.translate("MindMove", u"Create Training Dataset", None))
        self.trainingCreateDatasetsSelectRecordingsPushButton.setText(QCoreApplication.translate("MindMove", u"Select Recordings", None))
        self.trainingCreateDatasetPushButton.setText(QCoreApplication.translate("MindMove", u"Create Dataset", None))
        self.label_6.setText(QCoreApplication.translate("MindMove", u"Label", None))
        self.trainingTrainModelGroupBox.setTitle(QCoreApplication.translate("MindMove", u"Train Model", None))
        self.trainingTrainNewModelRadioButton.setText(QCoreApplication.translate("MindMove", u"New", None))
        self.trainingTrainModelPushButton.setText(QCoreApplication.translate("MindMove", u"Train", None))
        self.trainingSelectDatasetPushButton.setText(QCoreApplication.translate("MindMove", u"Select Dataset", None))
        self.trainingSelectedDatasetLabel.setText(QCoreApplication.translate("MindMove", u"Placeholder", None))
        self.trainingLoadExistingModelPushButton.setText(QCoreApplication.translate("MindMove", u"Load Model", None))
        self.trainingLoadExistingModelLabel.setText(QCoreApplication.translate("MindMove", u"Placeholder", None))
        self.label_8.setText(QCoreApplication.translate("MindMove", u"Label", None))
        self.trainingTrainExistingModelRadioButton.setText(QCoreApplication.translate("MindMove", u"Existing", None))
        self.onlineLoadModelGroupBox.setTitle(QCoreApplication.translate("MindMove", u"Load Model", None))
        self.onlineLoadModelPushButton.setText(QCoreApplication.translate("MindMove", u"Select Model", None))
        self.onlineModelLabel.setText(QCoreApplication.translate("MindMove", u"Placeholder", None))
        self.onlineCommandsGroupBox.setTitle(QCoreApplication.translate("MindMove", u"Commands", None))
        self.onlineRecordTogglePushButton.setText(QCoreApplication.translate("MindMove", u"Start Recording", None))
        self.protocolModeGroupBox.setTitle(QCoreApplication.translate("MindMove", u"Mode", None))
        self.protocolTrainingRadioButton.setText(QCoreApplication.translate("MindMove", u"Training", None))
        self.protocolOnlineRadioButton.setText(QCoreApplication.translate("MindMove", u"Online", None))
        self.protocolRecordRadioButton.setText(QCoreApplication.translate("MindMove", u"Record", None))
        self.mindMoveTabWidget.setTabText(self.mindMoveTabWidget.indexOf(self.procotolWidget), QCoreApplication.translate("MindMove", u"Protocol", None))
    # retranslateUi

