import sys
#from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog

from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *

from first import *

import cv2


class MyWindow(QMainWindow, Ui_ws_mainWindow):
    def __init__(self, parent=None):
        super(MyWindow, self).__init__(parent)
        self.setupUi(self)

        self.timer_camera = QTimer(self)
        self.cap = cv2.VideoCapture(0)
        self.timer_camera.timeout.connect(self.show_pic)
        self.timer_camera.start(10)
        
    # 调用打开文件的dialog. 
    def slot_openFile(self):
        filename, tmp = QFileDialog.getOpenFileName(
            self, '打开文件', '../../wsdata/', '*.avi *.mp4')

        self.filename = filename
        
        #self.cap = cv2.VideoCapture(filename)

    # 显示图像
    def show_pic(self):

        # 根据显示label的大小尺寸缩放图像
        w = self.label_show.width()
        h = self.label_show.height()

        success, frame = self.cap.read()
        if success:
            show = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            showImage = QImage(show.data, show.shape[1], show.shape[0], QImage.Format_RGB888)
            self.label_show.setPixmap(QPixmap.fromImage(showImage))
            self.timer_camera.start(10)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    myWin = MyWindow()
    myWin.show()
    sys.exit(app.exec_())
