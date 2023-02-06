import csv
from glob import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
#from PIL import ImageFile
#ImageFile.LOAD_TRUNCATED_IMAGES = True
RemakeLIST=0
DoProcess=1
SortUnlabled=0
DatasetRESOLUTION=None #dont change if None
labellinewidth =15

def MakeMarker(PairList,ShowImg=0,ReSize=1,Savedir=''):
    listfile=PairList[1]
    img = plt.imread(PairList[0])
    emptyimg = np.zeros(np.shape(img), dtype=np.uint8)
    exist_list=[]
    with open(listfile) as f:
        for line in f:
            line = line.strip()
            l = line.split(" ")
            exist_list.append([int(eval(x)) for x in l[2:]])
    lines_x = [np.array(exist_list[i])[np.arange(0, len(exist_list[i]), 2)] for i in range(len(exist_list))]
    lines_y = [np.array(exist_list[i])[np.arange(1, len(exist_list[i]), 2)] for i in range(len(exist_list))]
    for j in range(0, exist_list.__len__()):
        for i in range(len(lines_x[j]) - 1):
            # image = cv2.circle(img, (exist_list[j][i],exist_list[j][i+1]), radius=0, color=(0, 255, 0), thickness=10)
            linesimg = cv2.line(emptyimg , (lines_x[j][i], lines_y[j][i]), (lines_x[j][i + 1], lines_y[j][i + 1]),
                             color=(255, 255, 255), thickness=labellinewidth)
            #img = image #replace pixels?
    if ReSize:
        scale_percent=341/590
        width = int(img.shape[1] * scale_percent)
        height = int(img.shape[0] * scale_percent)
        if DatasetRESOLUTION!=None:
            imgs = cv2.resize(img, dsize=DatasetRESOLUTION)
            linesimgs = cv2.resize(linesimg, dsize=DatasetRESOLUTION)
        else:
            imgs=img
            linesimgs=linesimg
        #imgs = imgs[:, 192:-1]
        #imgs = imgs[:, 0:-192]
        #linesimgs = linesimgs[:, 192:-1]
        #linesimgs = linesimgs[:, 0:-192]

        #plt.imshow(imgs)
        #plt.show()

        img=imgs
        linesimg=linesimgs
    if ShowImg:
        fig, axs = plt.subplots(3)
        fig.suptitle('plots')
        axs[0].imshow(img)
        axs[1].imshow(cv2.addWeighted(linesimg, 1, img, 1, 0))
        axs[2].imshow(linesimg)
    # axs[0].set_title("input")
    # axs[2].set_title("combine")
    # axs[4].set_title("labels")
        plt.show()


    #lets generate names for images:
    tmp = listfile.split('/')
    name = tmp[-3].split("frame")[0] + '_' + tmp[-2] + '_' + tmp[-1].split('.lines.txt')[0]
    saveimgpath=Savedir+name+'_input.jpg'
    savelabelpath=Savedir+name+'_label.jpg'
    cv2.imwrite(saveimgpath,img)
    cv2.imwrite(savelabelpath,linesimg)
    print("end")
    return name


def Scalencrop(target,name,savepath):
    try:
        img = plt.imread(target)
    except:
        print("error:",target)
        return []
    scale_percent = 341 / 590
    width = int(img.shape[1] * scale_percent)
    height = int(img.shape[0] * scale_percent)
    imgs = cv2.resize(img, dsize=(width, height))
    imgs = imgs[:, 192:-1]
    imgs = imgs[:, 0:-192]
    cv2.imwrite(savepath+name,imgs)
    return savepath+name


TargetDir=r'/media/vjekod/NewVolume/CuLane/prepared'
Name4AnnotationFolder="annotations_new"
AnnotationFolder=TargetDir+"/"+Name4AnnotationFolder

ListFiles=glob(TargetDir+'/*') #get all files in TargetDir:
ListFiles.remove(AnnotationFolder) #remove anotation folder from list

if RemakeLIST:
    ListAnnFiles=glob(AnnotationFolder+"/*")
    lines=list()
    for each in ListAnnFiles:
        for entry in glob(each+"/*"):
            for file in glob(entry+"/*"):
                file=file.split(".lines.txt")[0]
                file=file.split(AnnotationFolder)[-1]
                lines.append(file.split(".lines.txt")[0])

#lines=list()

    jpgs=list()

    for each in ListFiles:
        for entry in glob(each+'/*'):
            js=glob(entry+'/*.jpg')
            fjs=list()
            for jseach in js:
                tmp=jseach.split('.jpg')[0]
                tmp=tmp.split(TargetDir)[-1]
                fjs.append(tmp)
            lsl=list()
            jpgs = jpgs+fjs
            #lines= lines+ lsl

#lets find pairs:
#lines.sort()
#  jpgs.sort()

    pairs=list()
    NoMatchLines=list()
    flag=0
    while len(lines):
        for each in lines:
            #print(each)
            for entry in jpgs:
                if entry==each:
                    pair=[TargetDir+entry+".jpg",AnnotationFolder+each+".lines.txt"]
                    pairs.append(pair)
                    jpgs.remove(entry)
                    lines.remove(each)
                    #print(len(lines),len(jpgs),len(pairs))
                    break
#`
    with open('pairs.csv', 'w') as f:
        # using csv.writer method from CSV package
        write=csv.writer(f)
        write.writerows(pairs)

    with open('unpairedIMG.csv', 'w') as f:
        # using csv.writer method from CSV package
        write=csv.writer(f)
        write.writerows(jpgs)
else:
    with open('pairs.csv', newline='') as f:
        # using csv.writer method from CSV package
        read=csv.reader(f)
        pairs=list(read)
        #read.writerows(pairs)

    with open('unpairedIMG.csv', newline='') as f:
        # using csv.writer method from CSV package
        read=csv.reader(f)
        unpairdjpgs=list(read)[0]

if DoProcess:
    EmptyLabelList=list()
    ListOfProcessedFiles=list()
    i=0
    for each in pairs:
        exist_list=list()
        with open(each[1]) as f:
            for line in f:
                line = line.strip()
                l = line.split(" ")
                exist_list.append([int(eval(x)) for x in l[2:]])
        if len(exist_list):
            newfile=MakeMarker(each,Savedir='/media/vjekod/NewVolume/CuLane/processedFAT/')
            ListOfProcessedFiles.append(newfile)
            print(i/len(pairs))
        else:
            EmptyLabelList.append(each)
        i=i+1

    with open('processedlist.csv', 'w') as f:
        # using csv.writer method from CSV package
        write=csv.writer(f)
        write.writerows(ListOfProcessedFiles)

    with open('EmptyLabelList.csv', 'w') as f:
        # using csv.writer method from CSV package
        write = csv.writer(f)
        write.writerows(EmptyLabelList)
else:
    with open('processedlist.csv', newline='') as f:
        # using csv.writer method from CSV package
        read=csv.reader(f)
        ListOfProcessedFiles=list(read)[0]

    with open('EmptyLabelList.csv', newline='') as f:
        # using csv.writer method from CSV package
        read=csv.reader(f)
        EmptyLabelList=list(read)
if SortUnlabled:
    EmptyLabelListfix=list()
    for each in EmptyLabelList:
        #each = EmptyLabelList[0]
        #aa = each.split("['")[1]
        #aa = aa.split("',")[0]
        #aa = aa.split(TargetDir)[1]
        EmptyLabelListfix.append(each)
    #EmptyLabelListfix=EmptyLabelList
    for each in unpairdjpgs:
        EmptyLabelListfix.append(each + '.jpg')

    #scale down nad crop unlabeld data
    unlabledlist=list()
    for each in EmptyLabelListfix:
        curneme=each[0].split('/')[1::][-3]+'_'+each[0].split('/')[1::][-2]+'_'+each[0].split('/')[1::][-1]#each.split('/')[1::][0]+'_'+each.split('/')[1::][1]+'_'+each.split('/')[1::][2]
        target=TargetDir+'/'+curneme
        unlabledlist.append(Scalencrop(target=each,name=curneme,savepath = '/media/vjekod/NewVolume/CuLane/unlabled/'))

    with open('unlabledlist.csv', 'w') as f:
        # using csv.writer method from CSV package
        write = csv.writer(f)
        write.writerows(unlabledlist)
else:
    with open('unlabledlist.csv', newline='') as f:
        # using csv.writer method from CSV package
        read=csv.reader(f)
        unlabledlist=list(read)[0]

print("stop")