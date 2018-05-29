# coding=utf-8
import tensorflow as tf
import cv2
import pandas as pd
import numpy as np
import os
import shutil
import math
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
class  NeuralBPNet(QObject):
    signal_train=pyqtSignal(int,float,float)
    def __init__(self):
        super(NeuralBPNet, self).__init__()
        self.istrain=os.path.exists(r"ckpt/")
        self.modelfile=r"ckpt/"
        self.isload=0
        self.initialModle()

    def trainmodel(self,rightfile,wrongfile,epochs=100000):
        if self.istrain:
            self.clearFiles(self.modelfile)
        state=0
        rf=open(rightfile)
        wf=open(wrongfile)
        self.df = pd.read_csv(rf, index_col=[0])
        self.df2 = pd.read_csv(wf, index_col=[0])
        lenth1 = self.df.shape[0]
        lenth2 = self.df2.shape[0]
        data = np.zeros((lenth1 + lenth2, 6), dtype=np.float32)
        data[0:lenth1, 0:5] = self.df.values
        data[0:lenth1, 5] = 1
        data[lenth1:lenth1 + lenth2, 0:5] = self.df2.values
        data[lenth1:lenth1 + lenth2, 5] = -1
        train_indices = np.random.choice(len(data), round(len(data) * 0.6), replace=False)
        test_indices = np.array(list(set(range(len(data))) - set(train_indices)))

        self.train_x = data[train_indices, 0:5]
        self.train_y = data[train_indices, 5:6]
        self.test_x = data[test_indices, 0:5]
        self.test_y = data[test_indices, 5:6]

        self.maxacurate=0.0
        for i in range(epochs):
            self.sess.run(self.train_step, feed_dict={self.x_data: self.train_x, self.y_data: self.train_y})
            if i % 200 == 0:
                self.result = self.sess.run(self.loss, feed_dict={self.x_data: self.train_x, self.y_data: self.train_y})
                tacurate=self.sess.run(self.accuracy, feed_dict={self.x_data: self.test_x, self.y_data: self.test_y})
                self.signal_train.emit(i,self.result,tacurate)
                if tacurate>self.maxacurate:
                    self.saver.save(self.sess, 'ckpt/mnist.ckpt', global_step=i + 1)
                    self.maxacurate=tacurate
        state=1
        return state



    def clearFiles(self,tdir):
        if os.path.exists(tdir):
            for root,dir,file in os.walk(tdir):
                for name in file:
                    path=os.path.join(root,name)
                    os.remove(path)
            os.removedirs(tdir)

    def copymodel(self,tdir):
        tempdir=self.modelfile
        if not os.path.exists(tempdir):
            os.mkdir(tempdir)
        if  os.path.realpath(tdir)!=os.path.realpath(tempdir):
            for root, dir, file in os.walk(tdir):
                for name in file:
                    paths = os.path.join(root, name)
                    pathd=os.path.join(tempdir,name)
                    shutil.copyfile(paths,pathd)

    def initialModle(self):
        self.x_data = tf.placeholder(dtype=tf.float32, shape=[None, 5], name="input_x")
        self.y_data = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="input_y")
        self.weight1 = tf.Variable(tf.random_normal([5, 7]), name="lay1_weight")
        self.bais1 = tf.Variable(tf.random_normal([7]), name="lay1_bais")
        self.weight2 = tf.Variable(tf.zeros([7, 1]), name="lay2_weight")
        self.bais2 = tf.Variable(tf.random_normal([1]), name="lay2_bais")
        self.lay1 = tf.matmul(self.x_data, self.weight1) + self.bais1
        self.lay2 = tf.nn.sigmoid(self.lay1)
        self.output = tf.matmul(self.lay1, self.weight2) + self.bais2
        self.loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.y_data - self.output), reduction_indices=[1]))
        self.train_step = tf.train.GradientDescentOptimizer(0.006).minimize(self.loss)
        self.corelation = tf.abs(tf.sign(self.output) - self.y_data)
        self.accuracy = 1 - tf.reduce_mean(tf.cast(self.corelation, tf.float32)) / 2
        self.sess = tf.Session()
        self.saver = tf.train.Saver()
        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)


    def classify(self,data):
        result=0
        if self.isload:
            tv=self.sess.run(self.output,feed_dict={self.x_data:data})
            result=np.sign(tv[0][0])
        return result

    def loadModel(self,tdir):
        self.copymodel(tdir)
        ckpt = tf.train.get_checkpoint_state(self.modelfile)
        if ckpt is not None:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            self.isload=1
            return 1
        else:
            return 0


class picprocess(QObject):
    signal_picUpdated = pyqtSignal(int,str)
    def __init__(self):
        super(picprocess, self).__init__()
        self.mat1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        self.mat2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50))
        self.datadic=dict()
        self.datadic.__setitem__("energy",list())
        self.datadic.__setitem__("H", list())
        self.datadic.__setitem__("con", list())
        self.datadic.__setitem__("dim", list())
        self.datadic.__setitem__("core", list())
        self.picsNums=0

    def getAllUn(self,datadir):
        piclist = list()
        index=0
        self.cleardata()
        for (root, dirs, files) in os.walk(datadir):
            for filname in files:
                piclist.append(os.path.join(root, filname))
        self.picsNums=len(piclist)
        for it in piclist:

            self.picdeal(it)
            index=index+1
            self.signal_picUpdated.emit(index,it)


    def cleardata(self):
        for key,value in self.datadic.items():
            value.clear()

    def write(self,filename):
        data=pd.DataFrame(self.datadic)
        data.to_csv(path_or_buf=filename)

    def cv_imread(self,file_path):
        cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
        return cv_img

    def picdeal(self,picurl):
        filename=picurl
        img = self.cv_imread(filename)
        width = img.shape[0]
        height = img.shape[1]
        timg = np.zeros(img.shape, dtype=np.uint8)
        timg = img
        ret, bw = cv2.threshold(timg[:, :, 2], 0.6 * 255, 255, cv2.THRESH_BINARY_INV)
        img1 = np.zeros(img.shape, dtype=np.uint8)
        for i in range(3):
            img1[:, :, i] = np.multiply(np.uint8(bw / 255), img[:, :, i])
        img1 = cv2.medianBlur(img1, ksize=3)
        lab = cv2.cvtColor(img1, cv2.COLOR_RGB2Lab)
        myimg = np.zeros(bw.shape, dtype=np.uint8)
        myimg = np.multiply(np.uint8((bw / 255)), lab[:, :, 1])
        tmpimg = cv2.equalizeHist(myimg)
        img2 = cv2.morphologyEx(tmpimg, cv2.MORPH_OPEN, self.mat1)
        img3 = cv2.morphologyEx(img2, cv2.MORPH_CLOSE, self.mat1)
        img4 = cv2.morphologyEx(img3, cv2.MORPH_OPEN, self.mat2)
        img5 = cv2.morphologyEx(img4, cv2.MORPH_CLOSE, self.mat2)
        ret2, img6 = cv2.threshold(img5, 200, 255, cv2.THRESH_BINARY)
        showimg = img5
        tw = cv2.resize(showimg, (int(width / 16), int(height / 16)))
        templist=self.calculate(tw)
        self.datadic["H"].append(templist[0])
        self.datadic["con"].append(templist[1])
        self.datadic["core"].append(templist[2])
        self.datadic["dim"].append(templist[3])
        self.datadic["energy"].append(templist[4])
        rs=np.zeros((1,5),dtype=np.float32)
        for i in range(5):
             rs[0,i]=templist[i]
        return rs

    def calculate(self,mat):
        width = mat.shape[0]
        height = mat.shape[1]
        tempm = np.uint8(mat / 16)
        r = np.zeros((16, 16, 4), dtype=np.float32)
        for i in range(16):
            for j in range(16):
                for m in range(width):
                    for n in range(height):
                        dl = m - 1
                        dr = m + 1
                        du = n + 1
                        if (dr < width) and (tempm[m, n] == i) and (tempm[dr, n] == j):
                            r[i, j, 0] = r[i, j, 0] + 1
                        if (du < height) and (tempm[m, n] == i) and (tempm[m, du] == j):
                            r[i, j, 1] = r[i, j, 1] + 1
                        if (dr < width) and (du < height) and (tempm[m, n] == i) and (tempm[dr, du] == j):
                            r[i, j, 2] = r[i, j, 2] + 1
                        if (dl >= 0) and (du < height) and (tempm[m, n] == i) and (tempm[dl, du] == j):
                            r[i, j, 3] = r[i, j, 3] + 1
        resultlist = list()
        energe = np.zeros(4, dtype=np.float32)
        Con = np.zeros(4, dtype=np.float32)
        H = np.zeros(4, dtype=np.float32)
        Ax = np.zeros(4, dtype=np.float32)
        Ay = np.zeros(4, dtype=np.float32)
        deltaX = np.zeros(4, dtype=np.float32)
        deltaY = np.zeros(4, dtype=np.float32)
        Dim = np.zeros(4, dtype=np.float32)
        Cor = np.zeros(4, dtype=np.float32)
        for i in range(4):
            tm = r[:, :, i]
            sums = np.sum(tm)
            tm = tm / sums
            value = np.sum(np.square(tm))
            r[:, :, i] = tm
            energe[i] = value
        for i in range(4):
            value1 = 0
            value2 = 0
            valuej = 0
            valuek = 0
            value3 = 0
            for j in range(16):
                for k in range(16):
                    tv = r[j, k, i]
                    if tv != 0:
                        value1 = value1 - r[j, k, i] * math.log(r[j, k, i])
                    value2 = value2 + (j - k) ** 2 * tv
                    valuej = valuej + j * tv
                    valuek = valuek + k * tv
                    value3 = value3 + tv / (1 + (j - k) ** 2)
            H[i] = value1
            Con[i] = value2
            Dim[i] = value3
            Ax[i] = valuej
            Ay[i] = valuek
        for i in range(4):
            ux = Ax[i]
            uy = Ay[i]
            detax = 0
            detay = 0
            tcor = 0
            for j in range(16):
                for k in range(16):
                    tv = r[j, k, i]
                    detax = (j - ux) ** 2 * tv + detax
                    detay = (k - uy) ** 2 * tv + detay
                    tcor = tcor + j * k * tv
            deltaX[i] = detax
            deltaY[i] = detay
            Cor[i] = tcor
        Cor = Cor - np.multiply(ux, uy)
        temp = np.multiply(deltaX, deltaY)
        Cor = np.divide(Cor, temp)
        resultlist.append(np.average(H))
        resultlist.append(np.average(Con))
        resultlist.append(np.average(Cor))
        resultlist.append(np.average(Dim))
        resultlist.append(np.average(energe))
        return resultlist
