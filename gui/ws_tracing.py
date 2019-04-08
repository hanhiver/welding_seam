# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ws_tracing.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1200, 980)
        MainWindow.setMinimumSize(QtCore.QSize(1200, 980))
        MainWindow.setMaximumSize(QtCore.QSize(12000, 9800))
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.button_openFile = QtWidgets.QPushButton(self.centralwidget)
        self.button_openFile.setGeometry(QtCore.QRect(880, 70, 120, 50))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.button_openFile.sizePolicy().hasHeightForWidth())
        self.button_openFile.setSizePolicy(sizePolicy)
        self.button_openFile.setObjectName("button_openFile")
        self.graphicsView_center = QtWidgets.QGraphicsView(self.centralwidget)
        self.graphicsView_center.setGeometry(QtCore.QRect(10, 10, 800, 600))
        self.graphicsView_center.setObjectName("graphicsView_center")
        self.graphicsView_right = QtWidgets.QGraphicsView(self.centralwidget)
        self.graphicsView_right.setGeometry(QtCore.QRect(420, 630, 390, 300))
        self.graphicsView_right.setObjectName("graphicsView_right")
        self.lineEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit.setGeometry(QtCore.QRect(830, 10, 321, 41))
        self.lineEdit.setObjectName("lineEdit")
        self.graphicsView_left = QtWidgets.QGraphicsView(self.centralwidget)
        self.graphicsView_left.setGeometry(QtCore.QRect(10, 630, 390, 300))
        self.graphicsView_left.setObjectName("graphicsView_left")
        self.button_startTracing = QtWidgets.QPushButton(self.centralwidget)
        self.button_startTracing.setGeometry(QtCore.QRect(1040, 70, 120, 50))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.button_startTracing.sizePolicy().hasHeightForWidth())
        self.button_startTracing.setSizePolicy(sizePolicy)
        self.button_startTracing.setObjectName("button_startTracing")
        self.button_exitApp = QtWidgets.QPushButton(self.centralwidget)
        self.button_exitApp.setGeometry(QtCore.QRect(1060, 880, 120, 50))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.button_exitApp.sizePolicy().hasHeightForWidth())
        self.button_exitApp.setSizePolicy(sizePolicy)
        self.button_exitApp.setObjectName("button_exitApp")
        self.toolButton_selectFile = QtWidgets.QToolButton(self.centralwidget)
        self.toolButton_selectFile.setGeometry(QtCore.QRect(1160, 10, 31, 41))
        self.toolButton_selectFile.setObjectName("toolButton_selectFile")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1200, 49))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.button_exitApp.clicked.connect(MainWindow.close)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        MainWindow.setTabOrder(self.lineEdit, self.button_openFile)
        MainWindow.setTabOrder(self.button_openFile, self.button_startTracing)
        MainWindow.setTabOrder(self.button_startTracing, self.button_exitApp)
        MainWindow.setTabOrder(self.button_exitApp, self.graphicsView_left)
        MainWindow.setTabOrder(self.graphicsView_left, self.graphicsView_right)
        MainWindow.setTabOrder(self.graphicsView_right, self.graphicsView_center)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.button_openFile.setText(_translate("MainWindow", "打开文件"))
        self.button_startTracing.setText(_translate("MainWindow", "开始跟踪"))
        self.button_exitApp.setText(_translate("MainWindow", "退出"))
        self.toolButton_selectFile.setText(_translate("MainWindow", "..."))

