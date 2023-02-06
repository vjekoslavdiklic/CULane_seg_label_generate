import glob

import tensorflow
import numpy
import cv2
import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import matplotlib.pyplot as plt
#from numba import cuda
import gc
import csv
from tensorflow.python.client import device_lib
import random
try:
    from PIL import Image
except ImportError:
    import Image
#import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras import backend as K
from mymodels import get_model_m2, get_model_unet
#cuda.select_device(0)

class CustomAccuracy(tf.keras.losses.Loss):
  def __init__(self):
    super().__init__()
  def call(self, y_true, y_pred):
    mse = (tf.square((y_pred-y_true)))*y_true
    rmse = tf.math.sqrt(mse)
    return mse*1000
LOADMODEL=0
showplt=1
Path2LearnData=r'/media/vjekod/NewVolume/CuLane/ReadyFAT/learn/'
PAth2ValidData=r'/media/vjekod/NewVolume/CuLane/ReadyFAT/valid/'
ETS2img1=r'/home/vjekod/Desktop/CULane_seg_label_generate/ets2testdata/3ac216af_2022_11_03_10_57_07_60_front.jpg'
ETS2img2=r'/home/vjekod/Desktop/CULane_seg_label_generate/ets2testdata/68fcf324_2022_11_03_10_39_23_87_front.jpg'
sufix4data='img/'
sufix4label='label/'
onlyoriginals=1
datalist=list([ETS2img1,ETS2img2])
curesl=(400,400)

LearnDataPATHcsv=r'/media/vjekod/NewVolume/CuLane/ReadyFAT/LearnFull.csv'
ValidDataPATHcsv=r'/media/vjekod/NewVolume/CuLane/ReadyFAT/ValidFull.csv'

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

def GetData(datalist=None,bw=False,scale=False,resize=False, label = False, cls_lbl = 2):
    dlist=list()
    i=0
    for each in datalist:
        if resize==False:
            e1 = tf.keras.preprocessing.image.load_img(each, grayscale=bw)
        else:
            e1 = tf.keras.preprocessing.image.load_img(each, grayscale=bw, target_size=(resize[0],resize[1]))
        e2 = tf.keras.preprocessing.image.img_to_array(e1)
        if scale:
            if e2.max()!=0:
                e2 *= 1 / e2.max()
            else:
                e2=np.zeros(e2.shape)
                print("error")
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


    # Define the model.

def writecsvlog(inputdata,filepath,firstrownames=['loss','acc','val_loss','val_acc']):
    xlist = inputdata
    xx = np.transpose(
        np.vstack((np.array(xlist[0][:]),
                   np.array(xlist[1][:]),
                   np.array(xlist[2][:]),
                   np.array(xlist[3][:]))))
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(firstrownames)
        for each in xx:
            writer.writerow(each)

def Makeplot(model,Xval,Yval,showplot=False,curesl=(562,341),savepath=False):
    Xo = (Xval[2:3, :, :, :])
    Yo = (Yval[2:3, :, :, :])
    Ye = model(Xval[0:1, :, :, :])
    Yo = (Yval[2:3, :, :, :])
    Yo = np.array(Yo[0, :, :, :])
    Yo = Yo.astype('float32')
    Ye = np.array(Ye[0, :, :, :])
    Xoi = Xo[0, :, :, :].astype('float32')
    Xoi = cv2.resize(Xoi, curesl)
    if Yo.shape[2]>1:
        Yo=Yo[:,:,1]
    if Ye.shape[2] > 1:
        Ye = Ye[:, :, 1]
    Yoi = cv2.cvtColor(Yo, cv2.COLOR_GRAY2RGB)
    Yei = cv2.cvtColor(Ye, cv2.COLOR_GRAY2RGB)
    Yoi *= 255 / Yoi.max()
    # Yoi=Yoi.astype('int64')
    Yei *= 255 / Yei.max()
    # Yei = Yei.astype('int64')
    Yoi = cv2.resize(Yoi, curesl)
    Yei = cv2.resize(Yei, curesl)
    Yoi = Yoi.astype('int64')
    Yei = Yei.astype('int64')

    rc = Yoi[:, :, 2]
    gch = Yei[:, :, 1]
    # create empty image with same shape as that of src image
    combine = np.zeros(Yoi.shape)

    combine[:, :, 2] = rc
    combine[:, :, 1] = gch

    Yeitmp = np.zeros(Yoi.shape)
    Yeitmp[:, :, 1] = gch
    Yei = Yeitmp

    Yoitmp = np.zeros(Yoi.shape)
    Yoitmp[:, :, 2] = rc
    Yoi = Yoitmp

    Yoi = Yoi.astype('int64')
    Yei = Yei.astype('int64')
    Xoi = Xoi.astype('int64')

    #Xoi=cv2.resize(Xoi, curesl)


    pl1 = cv2.addWeighted(Xoi, 1, Yoi, 1, 0)
    pl2 = cv2.addWeighted(Xoi, 1, Yei, 1, 0)
    pl3 = cv2.addWeighted(Yoi, 1, Yei, 1, 0)


    fig, axs = plt.subplots(2, 2)
    fig.suptitle('plots')
    axs[0, 0].imshow(Xoi)
    axs[1, 0].imshow(combine)
    axs[0, 1].imshow(pl1)
    axs[1, 1].imshow(pl2)
    if savepath:
        plt.savefig(savepath)
    else:
        plt.show()

    return Xoi,combine,pl1,pl2

def MakeplotwithETSDATA(model,datalist,curesl=(200,200),showrez=(562,341),savepath=False):
    data = GetData(datalist=datalist, bw=False, scale=False, resize=curesl)
    y = model(data)
    X0 = cv2.resize(data[0, :, :, :].astype('float32'), showrez).astype('int64')
    X1 = cv2.resize(data[1, :, :, :].astype('float32'), showrez).astype('int64')
    Y0 = cv2.resize(np.array(y[0, :, :, :]).astype('float32'), showrez)
    Y1 = cv2.resize(np.array(y[1, :, :, :]).astype('float32'), showrez)
    if Y0.shape[2]>1:
        Y0=Y0[:,:,1]
    if Y1.shape[2]>1:
        Y1=Y1[:,:,1]
    fig, axs = plt.subplots(2, 2)
    fig.suptitle('plots')
    axs[0, 0].imshow(X0)
    axs[1, 0].imshow(Y0)
    axs[0, 1].imshow(X1)
    axs[1, 1].imshow(Y1)
    if savepath:
        plt.savefig(savepath)
    else:
        plt.show()

    return 1


imgslist,labslist=ListFromCSV(LearnDataPATHcsv,picksize=50000)
Valimgslist,Vallabslist=ListFromCSV(ValidDataPATHcsv,picksize=100)

#imgslist=imgslist[10000:]
#labslist=labslist[10000:]
if LOADMODEL:
    model=tf.keras.models.load_model(r'/home/vjekod/Desktop/CULane_seg_label_generate/Results/epoch-0451-val_loss-0.0311-val_acc-0.0012.hdf5')
else:
    #model=get_model_m2(width=341,height=562,depth=3)
    model = get_model_unet((400,400),2)

    model.compile(optimizer='rmsprop',#tf.keras.optimizers.Adam(learning_rate=1e-3),
                #loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),#tf.keras.losses.CategoricalCrossentropy(),#'categorical_crossentropy',#'binary_crossentropy',#mse
                loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                #metrics=[tf.keras.metrics.CathegoricalAccuracy()])
                metrics=["categorical_accuracy"])
model.summary()
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath='Results/'+"after_epoch-0451_"+"epoch-{epoch:04d}-val_loss-{val_loss:.4f}-val_acc-{val_categorical_accuracy:.4f}.hdf5",
    save_weights_only=False,
    monitor='val_loss',
    mode='min',
    save_best_only=True)

print(device_lib.list_local_devices())
size=1000
noi=np.floor(len(imgslist)/size)
a=list(range(0,len(imgslist),size))
a.append((len(imgslist)-a[len(a)-1])+a[len(a)-1])

Xval=GetData((Valimgslist[0:100]), bw=False,scale=False,resize=curesl)#.astype("float32")
Yval=GetData((Vallabslist[0:100]), bw=True, scale=True,resize=curesl, label=True)#,#resize=curesl)
#Yval = np.expand_dims(np.argmax(Yval, axis = -1), axis = -1).astype("uint8")
losslist=list()
vallosslist=list()
acclist=list()
valacclist=list()
curepoch=0
#random.seed(10)
for ix in range(0,2000):
    z=[list(x) for x in zip(imgslist, labslist)]
    #z=random.sample(z,10000)

    random.seed(10)
    for i in range(0,len(a)-1,1):
        if len(z)>(a[i + 1]-a[i]):
            #zx = (random.sample(z, a[i + 1]-a[i]))
            #zx=z[a[i],a[i + 1]]
            zx=z[a[i]:a[i + 1]]
        else:
            zx=z[(-1*size):]#if end then pick last n=size elemnets and run
        cimg=list()
        clab=list()
        for each in zx:
            #z.remove(each)
            cimg.append(each[0])
            clab.append(each[1])
        #print(datalist = imgslist(a(i):a(i + 1)
        X = GetData(cimg, bw=False,scale=False,resize=curesl)#.astype("float32")
        Y = GetData(clab, bw=True, scale=True,resize=curesl, label=True)#,#resize=curesl)
        #Y = np.expand_dims(np.argmax(Y, axis = -1), axis = -1).astype("uint8")
        #Yy = GetData((labslist[a[i]:a[i + 1] - 1]), bw=False, scale=False)
        if (i==0)and(LOADMODEL==0):
            epochnum=1
        else:
            epochnum=1
        h=model.fit(X, Y, epochs=curepoch + epochnum, batch_size=25,validation_data=(Xval, Yval),initial_epoch=curepoch,callbacks=[model_checkpoint_callback])
        del X,Y
        gc.collect()

        #log values:
        curepoch = curepoch + epochnum
        losslist+=(h.history['loss'])
        vallosslist+=(h.history['val_loss'])
        acclist+=(h.history['categorical_accuracy'])
        valacclist+=(h.history['val_categorical_accuracy'])
        loss=vallosslist[-1]
        acc=vallosslist[-1]
        name = 'Results/after_epoch-0451_' + str(ix) + "acc" + str(acc) + "ep_" + str(curepoch)
        #Makeplot(model, Xval, Yval, showplot=True)#, curesl=curesl)
        #MakeplotwithETSDATA(model, datalist, curesl=curesl)


    #name='Results/mym2_'+str(ix)+"acc"+str(acc)+"ep_"+str(curepoch)
        #Makeplot(model, Xval, Yval, showplot=True, savepath=name + 'data.png')  # , curesl=curesl)
        #MakeplotwithETSDATA(model, datalist, curesl=curesl, savepath=name + 'ets2.png')
        writecsvlog(inputdata=[losslist, acclist,vallosslist, valacclist],filepath=name+'.csv')
model.save(name)

print('imdone')