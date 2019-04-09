import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from first import *


class MyWindow(QMainWindow, Ui_ws_mainWindow):
    def __init__(self, parent=None):
        super(MyWindow, self).__init__(parent)
        self.setupUi(self)
		
	# 调用打开文件的dialog. 
	def slot_openFile():

		filename, tmp = QFileDialog.getOpenFileName(
			self, '打开文件', '../../wsdata/', '*.avi *.mp4')

		if filename is '':
			return

		self.capture = cv2.VideoCapture(filename)

	# 显示图像
	def refreshShow():

		# 根据显示label的大小尺寸缩放图像
		w = self.label_show.width()
		h = self.label_show.height()
		




if __name__ == '__main__':
    app = QApplication(sys.argv)
    myWin = MyWindow()
    myWin.show()
    sys.exit(app.exec_())
