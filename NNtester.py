import glob

import tensorflow
import numpy
import csv
import cv2
import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import matplotlib.pyplot as plt
from numba import cuda
import gc
import random
try:
    from PIL import Image
except ImportError:
    import Image
#import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras import backend as K
from PIL import ImageOps
#from NNPrepareAndRun import ListFromCSV
tf.config.set_visible_devices([], 'GPU')

def ListFromCSV(pathtocsv,picksize=0,randomseed=10):
    with open(pathtocsv, 'r') as file:
        reader = csv.reader(file)
        listofcsv = list(reader)

    images=list()
    labels=list()
    if picksize>0:
        random.seed(randomseed)
        targetlist=random.sample(listofcsv, picksize)
    else:
        targetlist=listofcsv
    for each in targetlist:
        images.append('/'.join(pathtocsv.split('/')[0:-1])+'/'+each[0])
        labels.append('/'.join(pathtocsv.split('/')[0:-1])+'/'+each[1])
    return images,labels

def GetDataGen(datalist,labellist, batch_size, img_shape = False,scale=False, bw=False,should_shuffle=False):
    # create empty batch
    batch_img = np.zeros((batch_size,img_shape[0],img_shape[1],3))
    batch_label = np.zeros((batch_size,img_shape[0],img_shape[1],2))

    index = 0
    #while True:
    img_list=datalist
    lab_list=labellist
    while True:
        for i in range(batch_size):
            img_name = img_list[index]
            if img_shape == False:
                i1 = tf.keras.preprocessing.image.load_img(img_list[index], grayscale=bw)
                l1 = tf.keras.preprocessing.image.load_img(lab_list[index], grayscale=bw)
            else:
                i1 = tf.keras.preprocessing.image.load_img(img_list[index], grayscale=bw, target_size=(img_shape[0], img_shape[1]))
                l1 = tf.keras.preprocessing.image.load_img(lab_list[index], grayscale=bw, target_size=(img_shape[0], img_shape[1]))

            i2 = tf.keras.preprocessing.image.img_to_array(i1)
            l2 =tf.keras.preprocessing.image.img_to_array(l1)
            if scale:
                i2 *= 1 / i2.max()
                l2 *= 1 / l2.max()
            i2 = i2.astype('int')
            l2 = l2.astype('int')



            a = l2[:, :, 0]
            b = 1-(a)

            batch_img[i] = i2
            batch_label[i,:,:,0]=b
            batch_label[i, :, :,1] = a

            index += 1
            if index == len(img_list):
                index = 0

        yield batch_img,batch_label




def GetData(datalist=None,bw=False,scale=False,resize=False, label = False,cls_lbl = 2):
    dlist=list()
    i=0
    for each in datalist:
        if resize==False:
            e1 = tf.keras.preprocessing.image.load_img(each, grayscale=bw)
        else:
            e1 = tf.keras.preprocessing.image.load_img(each, grayscale=bw, target_size=(resize[0],resize[1]))
        e2 = tf.keras.preprocessing.image.img_to_array(e1)
        if scale:
            e2 *= 1 / e2.max()
        e2=e2.astype('int')
        if label:
            new_dim = list(e2.shape)
            new_dim[2] = cls_lbl
            new_label = np.zeros(new_dim)
            for _cls_lbl in range(cls_lbl):
                _dims = np.where(e2 == _cls_lbl)
                new_label[_dims[0], _dims[1], _cls_lbl] = 1
            #new_label[np.where(e2 == 1), 1] = 1
            e2 = new_label
        dlist.append(e2)
        #print("progress:",i / len(datalist))
        i = i + 1
    Data = np.array(dlist)
    return Data


def CompPlot(X,Y,YE,numofplots='default',label='Plot'):

    if numofplots=='default':
        numofplots=4
    else:
        numofplots=X.shape[0]

    fig, axs = plt.subplots(3, numofplots, figsize=(16, 9))

    fig.suptitle(label)
    for i in range(0,numofplots):
        #for j in range(0,3):
        axs[0, i].imshow(X[i, :, :, :])
        axs[0, i].set_title('Image')
        axs[1, i].imshow(np.argmax(Y[i, :, :, :], axis=-1))
        axs[1, i].set_title('True')
        axs[2, i].imshow(np.argmax(YE[i, :, :, :], axis=-1))
        axs[2, i].set_title('Estimate')

    plt.show()

    return 1


curesl=(400,400)
showplt=1
Path2LearnData=r'/media/vjekod/UBUNTU 22_0/Culane/readyshort/learn/'
PAth2ValidData=r'/media/vjekod/UBUNTU 22_0/Culane/readyshort/valid'
sufix4data='img/'
sufix4label='label/'

#modelpath=r'/home/vjekod/Desktop/CULane_seg_label_generate/Results/epoch-0002-val_loss-0.3130-val_acc-0.9806.hdf5'
modelpath=r'/home/vjekod/Desktop/CULane_seg_label_generate/newest/run_allepoch-5425-val_loss-0.0606-val_acc-0.9794.hdf5'
#eurotruck data:
ETS2img1=r'/home/vjekod/Desktop/CULane_seg_label_generate/ets2testdata/3ac216af_2022_11_03_10_57_07_60_front.jpg'
ETS2img2=r'/home/vjekod/Desktop/CULane_seg_label_generate/ets2testdata/68fcf324_2022_11_03_10_39_23_87_front.jpg'
ETS2datalist=list([ETS2img1,ETS2img2])


#LearnDataPATHcsv=r'/media/vjekod/NewVolume/CuLane/ReadyFAT/LearnOrigOnly.csv'
#ValidDataPATHcsv=r'/media/vjekod/NewVolume/CuLane/ReadyFAT/ValidFull.csv'

#imgslist,labslist=ListFromCSV(LearnDataPATHcsv,picksize=10)
#Valimgslist,Vallabslist=ListFromCSV(ValidDataPATHcsv,picksize=10)


#Xtrain=GetData((imgslist), bw=False,scale=False,resize=curesl)#.astype("float32")
#Ytrain=GetData((labslist), bw=True, scale=True,resize=curesl, label=True)#,#resize=curesl)

#Xval=GetData((Valimgslist), bw=False,scale=False,resize=curesl)#.astype("float32")
#Yval=GetData((Vallabslist), bw=True, scale=True,resize=curesl, label=True)#,#resize=curesl)

XETS2=GetData((ETS2datalist), bw=False,scale=False,resize=curesl)#.astype("float32")

sufix4data='img/'
sufix4label='label/'
onlyoriginals=1
curesl=(400,400)


#datalist=list([ETS2img1,ETS2img2])
#lablist=list([ETS2img1,ETS2img2])
#traindatalist=list([NNimgT1,NNimgT1])
#trainlablist=list([NNimgT2,NNimgT2])



#data=GetData(datalist=datalist,bw=False,scale=False,resize=(400,400))
#Tb=GetDataGen(datalist=datalist,labellist=lablist,batch_size=1, img_shape=(400,400))


model=tf.keras.models.load_model(modelpath)

#Yevaltrain=model(Xtrain)
#Yevalval=model(Xval)
YevalETS2=model(XETS2)



fig, axs = plt.subplots(2, 2)
fig.suptitle('Plots on ETS2 argmax')
axs[0, 0].imshow(XETS2[0,:,:,:])
axs[1, 0].imshow(np.argmax(YevalETS2[0,:,:,:],axis=-1))
axs[0, 1].imshow(XETS2[1,:,:,:])
axs[1, 1].imshow(np.argmax(YevalETS2[1,:,:,:],axis=-1))
plt.show()



CompPlot(Xtrain,Ytrain,Yevaltrain,label='TrainData')
CompPlot(Xval,Yval,Yevalval,label='EvalData')

print("end")


#a=GetDataGen(datalist=imgslist,labellist=labslist,batch_size=5,img_shape=curesl)

#for i in a:
 #   print(i)