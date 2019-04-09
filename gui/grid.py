# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'grid.ui'
#
# Created by: PyQt5 UI code generator 5.12.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_dialog(object):
    def setupUi(self, dialog):
        dialog.setObjectName("dialog")
        dialog.resize(1200, 660)
        self.lineEdit_filename = QtWidgets.QLineEdit(dialog)
        self.lineEdit_filename.setGeometry(QtCore.QRect(820, 10, 321, 42))
        self.lineEdit_filename.setObjectName("lineEdit_filename")
        self.button_openFile = QtWidgets.QToolButton(dialog)
        self.button_openFile.setGeometry(QtCore.QRect(1150, 10, 44, 41))
        self.button_openFile.setObjectName("button_openFile")
        self.button_exit = QtWidgets.QPushButton(dialog)
        self.button_exit.setGeometry(QtCore.QRect(1020, 590, 160, 50))
        self.button_exit.setObjectName("button_exit")
        self.button_startTrace = QtWidgets.QPushButton(dialog)
        self.button_startTrace.setGeometry(QtCore.QRect(840, 70, 160, 50))
        self.button_startTrace.setObjectName("button_startTrace")
        self.button_stopTrace = QtWidgets.QPushButton(dialog)
        self.button_stopTrace.setGeometry(QtCore.QRect(1020, 70, 160, 50))
        self.button_stopTrace.setObjectName("button_stopTrace")
        self.label_show = QtWidgets.QLabel(dialog)
        self.label_show.setGeometry(QtCore.QRect(10, 10, 800, 640))
        self.label_show.setFrameShadow(QtWidgets.QFrame.Plain)
        self.label_show.setText("")
        self.label_show.setObjectName("label_show")
        self.graphicsView_show = QtWidgets.QGraphicsView(dialog)
        self.graphicsView_show.setGeometry(QtCore.QRect(870, 250, 256, 192))
        self.graphicsView_show.setObjectName("graphicsView_show")

        self.retranslateUi(dialog)
        self.button_exit.clicked.connect(dialog.close)
        QtCore.QMetaObject.connectSlotsByName(dialog)

    def retranslateUi(self, dialog):
        _translate = QtCore.QCoreApplication.translate
        dialog.setWindowTitle(_translate("dialog", "焊缝跟踪"))
        self.button_openFile.setText(_translate("dialog", "..."))
        self.button_exit.setText(_translate("dialog", "退出"))
        self.button_startTrace.setText(_translate("dialog", "开始跟踪"))
        self.button_stopTrace.setText(_translate("dialog", "停止跟踪"))


