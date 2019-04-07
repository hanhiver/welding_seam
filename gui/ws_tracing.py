# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ws_tracing.ui'
#
# Created by: PyQt5 UI code generator 5.12.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(1200, 944)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(Dialog.sizePolicy().hasHeightForWidth())
        Dialog.setSizePolicy(sizePolicy)
        Dialog.setMinimumSize(QtCore.QSize(1200, 944))
        Dialog.setMaximumSize(QtCore.QSize(1200, 944))
        self.graphicsView = QtWidgets.QGraphicsView(Dialog)
        self.graphicsView.setGeometry(QtCore.QRect(20, 20, 800, 900))
        self.graphicsView.setObjectName("graphicsView")
        self.lineEdit = QtWidgets.QLineEdit(Dialog)
        self.lineEdit.setGeometry(QtCore.QRect(840, 20, 341, 41))
        self.lineEdit.setObjectName("lineEdit")
        self.button_startTracing = QtWidgets.QPushButton(Dialog)
        self.button_startTracing.setGeometry(QtCore.QRect(1020, 80, 120, 50))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.button_startTracing.sizePolicy().hasHeightForWidth())
        self.button_startTracing.setSizePolicy(sizePolicy)
        self.button_startTracing.setObjectName("button_startTracing")
        self.button_openFile = QtWidgets.QPushButton(Dialog)
        self.button_openFile.setGeometry(QtCore.QRect(880, 80, 120, 50))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.button_openFile.sizePolicy().hasHeightForWidth())
        self.button_openFile.setSizePolicy(sizePolicy)
        self.button_openFile.setObjectName("button_openFile")

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "焊缝跟踪"))
        self.button_startTracing.setText(_translate("Dialog", "开始跟踪"))
        self.button_openFile.setText(_translate("Dialog", "打开文件"))


