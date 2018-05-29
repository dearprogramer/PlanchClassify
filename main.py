# coding=utf-8

import sys
import Gui
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *


if __name__ == "__main__":
    app = QApplication(sys.argv)    #创建QApplication类的实例
    dia =Gui.MainWin()              #创建DumbDialog类的实例
    dia.show()                      #显示程序主窗口
    app.exec_()

