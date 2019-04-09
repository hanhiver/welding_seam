import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (QApplication, QDialog, QFileDialog, QGridLayout,
                             QLabel, QPushButton)
from grid import *

import cv2


class MyWindow(QMainWindow, Ui_dialog):
    def __init__(self, parent=None):
        super(MyWindow, self).__init__(parent)
        self.setupUi(self)

    def slot_openFile(self):
        # 调用打开文件的dialog.
        fileName, tmp = QFileDialog.getOpenFileName(
            self, '打开文件', '../', '*.png *.jpg *.bmp')

        if fileName is '':
            return

        self.img = cv2.imread(fileName)

        if self.img.size == 1:
            return

        self.refreshShow() 

    def refreshShow(self):
        # 根据label_show尺寸缩放图像.
        width = self.label_show.width() 
        height = self.label_show.height()
        self.img = cv2.resize(self.img, (height, width))

        # 提取图形尺寸和通道，将opencv下的img转换成Qimage
        height, width, channel = self.img.shape
        bytesPerLine = 3 * width
        self.qImg = QImage(self.img.data, width, height, bytesPerLine, 
                           QImage.Format_RGB888).rgbSwapped()

        # 将图像显示出来
        self.label_show.setPixmap(QPixmap.fromImage(self.qImg))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    myWin = MyWindow()
    myWin.button_openFile.clicked.connect(myWin.slot_openFile)
    myWin.show()
    sys.exit(app.exec_())
