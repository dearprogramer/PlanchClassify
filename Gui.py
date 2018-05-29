# coding=utf-8
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import cv2
import demo
import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Qt5Agg")  # 声明使用QT5
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
class MyMplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)
        self.axes = self.fig.add_subplot(111)

    def getmin(self,a, b):
        if a > b:
            return b
        else:
            return a

    def getmax(self,a, b):
        if a < b:
            return b
        else:
            return a

    def drawPic(self,data, title):
        t=data[:,0]
        data1=data[:,1]
        data2=data[:,2]
        color = 'tab:red'
        self.axes.set_xlim([0,600])
        self.axes.set_xlabel('epoch (200)')
        self.axes.set_ylabel('loss', color=color)
        self.axes.plot(t, data1, color=color)
        self.axes.tick_params(axis='y', labelcolor=color)
        ax2 = self.axes.twinx()  # instantiate a second axes that shares the same x-axis
        color = 'tab:blue'
        ax2.set_ylabel('acurrate rate', color=color)  # we already handled the x-label with ax1
        ax2.plot(t, data2, color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        self.axes.set_title("loss and acurate graph")



class MainOp(QThread):
    signal_finished=pyqtSignal(int,str)
    signal_stoped=pyqtSignal(int)
    def __init__(self):
        super(MainOp,self).__init__()
        self.isload=0
        self.bpn =demo.NeuralBPNet()
        self.picdir=""
        self.actionType=0
        self.traindata=list()
        self.picd=demo.picprocess()
        self.picfile="1.jpg"
        self.singlePic=""
        self.backimg="1.jpg"
        self.resulutdata=None
        self.picNums=0
        self.count=0

    def loadmodel(self,dir):
        self.isload=self.bpn.loadModel(dir)
        return self.isload

    def classify(self,file):
        result=0
        if self.isload:
            self.actionType=4
            self.singlePic=file
            self.start()
            result=1
        return result


    def calculateTruedata(self,dir):
        self.actionType=1
        self.picdir=dir
        self.start()

    def calculateFalsedata(self,dir):
        self.actionType=2
        self.picdir=dir
        self.start()

    def trianmodel(self,turefile,falsefile):
        self.truefile=turefile
        self.falsefile=falsefile
        self.actionType=3
        self.start()

    def run(self):
        if self.actionType==1:
            self.picd.signal_picUpdated.connect(self.picProcessStop)
            self.picd.getAllUn(self.picdir)
            self.picd.write("true.csv")
            msg = "结果： 对膨大果进行处理，图片文件夹：" + self.picdir + " 结果文件：true.csv"
            self.signal_finished.emit(self.actionType,msg)

        if self.actionType==2:
            self.picd.signal_picUpdated.connect(self.picProcessStop)
            self.picd.getAllUn(self.picdir)
            self.picd.write("false.csv")
            msg="结果： 对膨大果进行处理，图片文件夹："+self.picdir+" 结果文件：false.csv"
            self.signal_finished.emit(self.actionType,msg)

        if self.actionType==3:
            self.count=0
            self.bpn.signal_train.connect(self.trian_stop)
            self.isload=self.bpn.trainmodel(self.truefile,self.falsefile)
            msg="结果：  准确度:"+str(self.bpn.maxacurate)+" 损失:"+str(self.bpn.result)
            data=np.array(self.traindata)
            print(data)
            self.signal_finished.emit(self.actionType, msg)

        if self.actionType==4:
            self.picd.cleardata()
            tempdata = self.picd.picdeal(self.singlePic)
            print("here1")
            result = self.bpn.classify(tempdata)
            print("here2:"+str(result))
            msg="错误："
            if result==1:
                msg="结果： 图像为未膨大果(正常)"
            if result==-1:
                msg="结果： 图像为膨大果(异常)"
            self.signal_finished.emit(self.actionType, msg)

    def picProcessStop(self,index,picurl):
        self.picNums=self.picd.picsNums
        self.picindex=index
        self.picfile=picurl
        self.signal_stoped.emit(self.actionType)



    def trian_stop(self,epoch,loss,accurate):
        tepoch=int(epoch/200)
        tl=list()
        tl.append(tepoch)
        tl.append(loss)
        tl.append(accurate)
        self.count = self.count + 1
        self.traindata.append(tl)
        if (self.count+1)%10==0:
            self.resulutdata=np.array(self.traindata)
            self.resulutdata=self.resulutdata.reshape((-1,3))
            self.signal_stoped.emit(self.actionType)

class MainWin(QMainWindow):
    def __init__(self,parent=None):
        super(MainWin, self).__init__(parent)
        self.picwidth=600
        self.picheight=400
        self.isload=0
        self.modledir = ""
        self.background="back.png"
        self.turefiles="true.csv"
        self.falsefiles="false.csv"
        self.order=0
        self.isexcute=0
        self.istestmode=0
        self.Mo=MainOp()
        self.picfile=""
        self.initialMainframe()

    def initialMainframe(self):
        self.image = QImage()
        self.dirty = False
        self.filename = None
        self.setWindowTitle("猕猴桃识别—TY && LCW")
        self.mirroredvertically = False
        self.mirroredhorizontally = False
        self.reserveinfoitems = 0
        self.menu = QMenuBar()
        self.picMenu=self.menu.addMenu("图片")
        self.batdeal=self.picMenu.addMenu("批处理")
        self.dealTrues=self.batdeal.addAction("非膨大果")
        self.dealFalse=self.batdeal.addAction("膨大果")
        self.picRecognize=self.picMenu.addAction("识别")
        self.modelMenu=self.menu.addMenu("模型")
        self.modelLoad=self.modelMenu.addAction("载入")
        self.modeltrain=self.modelMenu.addAction("训练")
        self.excuteAction=QPushButton("执行")
        self.excuteParame=QTextEdit()
        self.excuteParame.setText("警告：模型未加载，不能识别")
        self.excuteParame.setReadOnly(True)
        self.resultParame=QTextEdit()
        self.resultParame.setText("结果")
        self.resultParame.setReadOnly(True)
        self.canval=QGraphicsView()
        self.scen=QGraphicsScene()
        self.grapscen=QGraphicsScene()
        self.canval.setScene(self.scen)
        self.excuteAction.clicked.connect(self.excute)
        self.modelLoad.triggered.connect(self.loadModel)
        self.picRecognize.triggered.connect(self.classifypic)
        self.dealFalse.triggered.connect(self.falsePicsBatDeal)
        self.dealTrues.triggered.connect(self.turePicsBatDeal)
        self.modeltrain.triggered.connect(self.train)
        self.Mo.signal_finished.connect(self.resultdisplay)
        self.Mo.signal_stoped.connect(self.updatePic)
        self.conentFrame=QFrame()
        self.total_Vlayout=QVBoxLayout()
        self.picLable=QLabel()
        self.displayImage(self.background)
        self.canvalinit(self.background)
        self.mid_Hlayout=QHBoxLayout()
        self.mid_Hlayout.addWidget(self.excuteParame)
        self.mid_Hlayout.addWidget(self.excuteAction)
        self.setMenuBar(self.menu)
        self.total_Vlayout.addWidget(self.canval)
        self.total_Vlayout.addLayout(self.mid_Hlayout)
        self.total_Vlayout.addWidget(self.resultParame)
        self.conentFrame.setLayout(self.total_Vlayout)
        self.setCentralWidget(self.conentFrame)

    def updatePic(self,type):
        if type==3:
            tempdata=np.zeros(self.Mo.resulutdata.shape)
            tempdata=self.Mo.resulutdata[:,:]
            print(tempdata)
            self.drawGragh(tempdata,"损失图")
        if type==2:
            print("display pic")
            print(self.Mo.picfile)
            self.displayImage(self.Mo.picfile)
            self.resultParame.setText("结果： 当前处理第" + str(self.Mo.picindex) + "张图片 ，总共" + str(self.Mo.picNums) + "张图片")
        if type==1:
            print("display pic")
            print(self.Mo.picfile)
            self.displayImage(self.Mo.picfile)
            self.resultParame.setText("结果： 当前处理第"+str(self.Mo.picindex)+"张图片 ，总共"+str(self.Mo.picNums)+"张图片")




    def canvalinit(self,filename):
        picdata = self.cv_imread(filename)
        picdata = cv2.cvtColor(picdata, cv2.COLOR_BGR2RGB)
        self.backpic = cv2.resize(picdata, (self.picwidth, self.picheight))
        self.pic = QImage(self.backpic.data,self.picwidth, self.picheight, QImage.Format_RGB888)
        self.scen.addPixmap(QPixmap.fromImage(self.pic))

    def cv_imread(self,file_path):
        cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
        return cv_img

    def excute(self):
        if not self.istestmode:
            if self.order:
                self.excuteState()
                if self.order==1:
                    self.isload=self.Mo.loadmodel(self.modledir)
                    self.normalState()
                if self.order==2:
                    tr=self.Mo.classify(self.picfile)
                    if tr==0: #如果执行异常
                        self.normalState()
                if self.order==3:
                    self.Mo.trianmodel(self.turefiles,self.falsefiles)
                if self.order==4:
                    self.Mo.calculateTruedata(self.turePics)
                if self.order==5:
                    self.Mo.calculateFalsedata(self.falsePics)

            else:
                QMessageBox.about(self, '错误!', '请先选择将要执行的操作!')
        else:
            if self.istestmode:
                self.exuteTest()
                self.normalState()

    def resultdisplay(self,type,msg):
        self.resultParame.setText(msg)
        self.normalState()
    #
    def exuteTest(self):
        picdir=r"pic/"
        self.Mo.calculateFalsedata(picdir)

    def excuteState(self):
         self.isexcute=1
         self.excuteAction.setText("处理中")
         print("处理中")
         self.excuteAction.setEnabled(False)

    def normalState(self):
        self.order = 0
        self.isexcute=0
        self.excuteAction.setText("执行")
        self.excuteAction.setEnabled(True)
        print("执行")
        if not self.isload:
            self.excuteParame.setText("警告：模型未加载，不能识别")
        else:
            self.excuteParame.setText("模型已加载，可以识别")

    def classifypic(self):
        if not self.isexcute:
            if self.isload:
                tempfile=QFileDialog.getOpenFileName(self, '选取图片', './',"Image files(*.jpg)")
                self.picfile = tempfile[0]
                if self.picfile != "":
                    self.excuteParame.setText("将操作：识别图像 图片位置：" + self.picfile)
                    self.displayImage(self.picfile)
                    self.order=2


    def loadModel(self):
        if not self.isexcute:
            self.modledir=self.getdir("选取模型文件夹")
            if self.modledir!="":
                self.order=1
                self.excuteParame.setText("将操作：加载模型 模型位置：" + self.modledir)


    def turePicsBatDeal(self):
        if not self.isexcute:
            self.turePics=self.getdir("选取保存未膨大果的文件夹（正常）")
            if self.turePics!=None:
                self.excuteParame.setText("将操作：批处理图形非膨大果 图片位置："+self.turePics)
                self.order = 4

    def falsePicsBatDeal(self):
        if not self.isexcute:
            self.falsePics=self.getdir("选取保存膨大果的文件夹（异常）")
            if self.turePics!="":
                self.excuteParame.setText("将操作：批处理膨大果 图片位置："+self.falsePics)
                self.order=5
            print(self.falsePics)


    def getdir(self,title):
        dir=QFileDialog.getExistingDirectory(self,title, './')
        dir=dir+r"/"
        return dir

    def train(self):
        if not self.isexcute:
            self.turefiles=QFileDialog.getOpenFileName(self, '非膨大果数据(正常)', './',"Image files(*.csv)")[0]
            self.falsefiles=QFileDialog.getOpenFileName(self, '膨大果数据（异常）', './',"Image files(*.csv)")[0]
            if self.turefiles!="" and self.falsefiles!="":
                self.excuteParame.setText("将操作：训练模型 非膨大果数据：" + self.turefiles+" 膨大果数据:"+self.falsefiles)
                self.order=3

    def displayImage(self,filename):
        picdata = self.cv_imread(filename)
        picdata = cv2.cvtColor(picdata, cv2.COLOR_BGR2RGB)
        picdata= cv2.resize(picdata, (self.picwidth, self.picheight))
        tempic = QImage(picdata,self.picwidth, self.picheight, QImage.Format_RGB888)
        self.scen.clear()
        self.scen.addPixmap(QPixmap.fromImage(tempic))
        self.canval.setScene(self.scen)
        self.canval.show()

    def drawGragh(self,data,title):
        self.grapscen.clear()
        tfigure = MyMplCanvas()
        tfigure.drawPic(data,title)
        self.grapscen.addWidget(tfigure)
        self.canval.setScene(self.grapscen)
        self.canval.show()




