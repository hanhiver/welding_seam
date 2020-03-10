# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'grid2.ui'
#
# Created by: PyQt5 UI code generator 5.12.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_dialog(object):
    def setupUi(self, dialog):
        dialog.setObjectName("dialog")
        dialog.resize(1007, 671)
        dialog.setWindowOpacity(1.0)
        self.horizontalLayout = QtWidgets.QHBoxLayout(dialog)
        self.horizontalLayout.setSpacing(9)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.frame_show = QtWidgets.QFrame(dialog)
        self.frame_show.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_show.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_show.setLineWidth(1)
        self.frame_show.setObjectName("frame_show")
        self.label_show = QtWidgets.QLabel(self.frame_show)
        self.label_show.setGeometry(QtCore.QRect(6, 6, 481, 641))
        self.label_show.setObjectName("label_show")
        self.horizontalLayout.addWidget(self.frame_show)
        self.frame_tools = QtWidgets.QFrame(dialog)
        self.frame_tools.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_tools.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_tools.setObjectName("frame_tools")
        self.button_startTrace = QtWidgets.QPushButton(self.frame_tools)
        self.button_startTrace.setGeometry(QtCore.QRect(50, 340, 226, 31))
        self.button_startTrace.setObjectName("button_startTrace")
        self.button_exit = QtWidgets.QPushButton(self.frame_tools)
        self.button_exit.setGeometry(QtCore.QRect(10, 480, 283, 31))
        self.button_exit.setObjectName("button_exit")
        self.button_stopTrace = QtWidgets.QPushButton(self.frame_tools)
        self.button_stopTrace.setGeometry(QtCore.QRect(40, 400, 379, 31))
        self.button_stopTrace.setObjectName("button_stopTrace")
        self.horizontalLayoutWidget = QtWidgets.QWidget(self.frame_tools)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(9, -1, 571, 31))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.hLayout_openFile = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.hLayout_openFile.setContentsMargins(0, 0, 0, 0)
        self.hLayout_openFile.setObjectName("hLayout_openFile")
        self.lineEdit_filename = QtWidgets.QLineEdit(self.horizontalLayoutWidget)
        self.lineEdit_filename.setObjectName("lineEdit_filename")
        self.hLayout_openFile.addWidget(self.lineEdit_filename)
        self.button_openFile = QtWidgets.QToolButton(self.horizontalLayoutWidget)
        self.button_openFile.setObjectName("button_openFile")
        self.hLayout_openFile.addWidget(self.button_openFile)
        self.horizontalLayout.addWidget(self.frame_tools)

        self.retranslateUi(dialog)
        self.button_exit.clicked.connect(dialog.close)
        QtCore.QMetaObject.connectSlotsByName(dialog)

    def retranslateUi(self, dialog):
        _translate = QtCore.QCoreApplication.translate
        dialog.setWindowTitle(_translate("dialog", "焊缝跟踪"))
        self.label_show.setText(_translate("dialog", "TextLabel"))
        self.button_startTrace.setText(_translate("dialog", "开始跟踪"))
        self.button_exit.setText(_translate("dialog", "退出"))
        self.button_stopTrace.setText(_translate("dialog", "停止跟踪"))
        self.button_openFile.setText(_translate("dialog", "..."))


