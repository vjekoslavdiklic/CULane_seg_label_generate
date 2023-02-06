import csv
from glob import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import shutil
from PIL import Image
SourcePath=r'/media/vjekod/NewVolume/CuLane/processedFAT/'
LearnDataPath=r'/media/vjekod/NewVolume/CuLane/ReadyFAT/learn/'
ValidDataPath=r'/media/vjekod/NewVolume/CuLane/ReadyFAT/valid/'
ImgFolder='img/'
LabFolder='label/'
ImgSufix='_input.jpg'
LabSufix='_label.jpg'
Makepremutations=1

def change_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.add(v,value)
    v[v > 255] = 255
    v[v < 0] = 0
    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

def RNDPerspectiveCrop(im,lb):
    dim = list(np.shape(im[:, :, 0]))
    w = dim[1]
    h = dim[0]
    corners = [[0, 0], [w, 0], [0, h], [w, h]]
    # random.randint(-1, 1)
    ax = [0, (np.floor(w * 1 / 4).astype('int'))]
    ay = [(np.floor(h * 1 / 3).astype('int')), (np.floor(h * 1 / 2).astype('int'))]

    bx = [(np.floor(w * 3 / 4).astype('int')), w]
    by = ay

    cx = ax
    cy = [(np.floor(h * 3 / 4).astype('int')), h]

    dx = bx
    dy = cy

    dots = [[random.randint(ax[0], ax[1]), random.randint(ay[0], ay[1])],
            [random.randint(bx[0], bx[1]), random.randint(by[0], by[1])],
            [random.randint(cx[0], cx[1]), random.randint(cy[0], cy[1])],
            [random.randint(dx[0], dx[1]), random.randint(dy[0], dy[1])]]
    dots = np.float32(dots)
    corners = np.float32(corners)

    pts1 = np.float32([[56, 65], [368, 52], [28, 387], [389, 390]])
    pts2 = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])

    M = cv2.getPerspectiveTransform(dots, corners)

    # M = cv2.getPerspectiveTransform(pts1,pts2)
    im = cv2.warpPerspective(im, M, (w, h))
    lb =cv2.warpPerspective(lb, M, (w, h))

    return im ,lb


Path2Processedimages=r'/media/vjekod/NewVolume/CuLane/processedFAT'
with open('processedlist.csv', newline='') as f:
    # using csv.writer method from CSV package
    read = csv.reader(f)
    ListOfProcessedFiles = list(read)
    ListOfProcessedFilesfix=list()
for each in ListOfProcessedFiles:
    each="".join(each)
    ListOfProcessedFilesfix.append(each)

ListOfProcessedFiles=ListOfProcessedFilesfix
nsum=len(ListOfProcessedFiles)
n4learn=int(np.ceil(nsum*0.8))
n4valid=nsum-n4learn

LearnList=random.sample(ListOfProcessedFiles, n4learn)
#remove Learnlist from main list
ValidList=ListOfProcessedFiles
for each in LearnList:
    ValidList.remove(each)

i=0
for each in LearnList:
    #shutil.copy(SourcePath + each + ImgSufix, LearnDataPath + ImgFolder + each + ImgSufix)
    #shutil.copy(SourcePath + each + LabSufix, LearnDataPath + LabFolder + each + LabSufix)

    curimg=cv2.imread(SourcePath + each + ImgSufix)
    curlab=cv2.imread(SourcePath + each + LabSufix)
    #br=increase_brightness(curimg, random.randint(-10, -1))
    if random.randint(0,1):
        br = change_brightness(curimg, random.randint(40, 80))
    else:
        br = change_brightness(curimg, random.randint(-90, -50))
    rotval=random.randint(-180, 180)
    rotimg = np.array((Image.fromarray(curimg).rotate(rotval)))
    rotlab = np.array((Image.fromarray(curlab).rotate(rotval)))
    flipval=random.randint(-1, 1)
    flipimg=cv2.flip(curimg, flipval)
    fliplab = cv2.flip(curlab, flipval)

    perimg,perlab=RNDPerspectiveCrop(curimg,curlab)
    #cv2.cvtColor(im_cv, cv2.COLOR_BGR2RGB)
    if (curlab.max()!=0)and(rotlab.max()!=0)and(fliplab.max()!=0)and(perlab.max()!=0):
        cv2.imwrite(LearnDataPath + ImgFolder + each + ImgSufix, cv2.cvtColor(curimg, cv2.COLOR_BGR2RGB))
        cv2.imwrite(LearnDataPath + LabFolder + each + LabSufix, curlab)

        if Makepremutations:
            cv2.imwrite(LearnDataPath + ImgFolder + each + "_bri_"  + ImgSufix, cv2.cvtColor(br, cv2.COLOR_BGR2RGB))
            cv2.imwrite(LearnDataPath + ImgFolder + each + "_rot_" + ImgSufix, cv2.cvtColor(rotimg, cv2.COLOR_BGR2RGB))
            cv2.imwrite(LearnDataPath + ImgFolder + each + "_flp_" + ImgSufix, cv2.cvtColor(flipimg, cv2.COLOR_BGR2RGB))
            cv2.imwrite(LearnDataPath + ImgFolder + each + "_per_" + ImgSufix, cv2.cvtColor(perimg, cv2.COLOR_BGR2RGB))
            cv2.imwrite(LearnDataPath + LabFolder + each + "_rot_" + LabSufix, rotlab)
            cv2.imwrite(LearnDataPath + LabFolder + each + "_flp_" + LabSufix, fliplab)
            cv2.imwrite(LearnDataPath + LabFolder + each + "_bri_" + LabSufix, curlab)
            cv2.imwrite(LearnDataPath + LabFolder + each + "_per_" + LabSufix, perlab)
    else:
        print("dropping, empty label",each)
    print('learn',i/len(LearnList))
    i=i+1

i=0
for each in ValidList:
    curimg = cv2.imread(SourcePath + each + ImgSufix)
    curlab = cv2.imread(SourcePath + each + LabSufix)
    if (curlab.max() != 0):
        cv2.imwrite(ValidDataPath + ImgFolder + each + ImgSufix, cv2.cvtColor(curimg, cv2.COLOR_BGR2RGB))
        cv2.imwrite(ValidDataPath + LabFolder + each + LabSufix, curlab)
    else:
        print("dropping, empty label",each)
    print('learn',i/len(ValidList))
    i=i+1


print("whereishere")