# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'first.ui'
#
# Created by: PyQt5 UI code generator 5.12.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_ws_mainWindow(object):
    def setupUi(self, ws_mainWindow):
        ws_mainWindow.setObjectName("ws_mainWindow")
        ws_mainWindow.resize(1000, 700)
        self.label_show = QtWidgets.QLabel(ws_mainWindow)
        self.label_show.setGeometry(QtCore.QRect(10, 10, 800, 640))
        self.label_show.setObjectName("label_show")
        self.verticalLayoutWidget = QtWidgets.QWidget(ws_mainWindow)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(819, 10, 171, 671))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setSpacing(20)
        self.verticalLayout.setObjectName("verticalLayout")
        self.pushButton_openFile = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.pushButton_openFile.setObjectName("pushButton_openFile")
        self.verticalLayout.addWidget(self.pushButton_openFile)
        self.pushButton_startTracing = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.pushButton_startTracing.setObjectName("pushButton_startTracing")
        self.verticalLayout.addWidget(self.pushButton_startTracing)
        self.pushButton_stopTracing = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.pushButton_stopTracing.setObjectName("pushButton_stopTracing")
        self.verticalLayout.addWidget(self.pushButton_stopTracing)
        self.pushButton_exit = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.pushButton_exit.setObjectName("pushButton_exit")
        self.verticalLayout.addWidget(self.pushButton_exit)

        self.retranslateUi(ws_mainWindow)
        self.pushButton_exit.clicked.connect(ws_mainWindow.close)
        QtCore.QMetaObject.connectSlotsByName(ws_mainWindow)

    def retranslateUi(self, ws_mainWindow):
        _translate = QtCore.QCoreApplication.translate
        ws_mainWindow.setWindowTitle(_translate("ws_mainWindow", "Dialog"))
        self.label_show.setText(_translate("ws_mainWindow", "TextLabel"))
        self.pushButton_openFile.setText(_translate("ws_mainWindow", "Open File"))
        self.pushButton_startTracing.setText(_translate("ws_mainWindow", "Start Tracing"))
        self.pushButton_stopTracing.setText(_translate("ws_mainWindow", "Stop Tracing"))
        self.pushButton_exit.setText(_translate("ws_mainWindow", "Exit"))


