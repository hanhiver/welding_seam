import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from ws_tracing import *


class MyWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MyWindow, self).__init__(parent)
        self.setupUi(self)

def openFile():
	filename = getOpenFileName()
	return filename

if __name__ == '__main__':
    app = QApplication(sys.argv)
    myWin = MyWindow()
    myWin.button_openFile.clicked.connect(openFile) 
    myWin.show()
    sys.exit(app.exec_())
