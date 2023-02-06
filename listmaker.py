import glob
import csv
Path2LearnData=r'/media/vjekod/NewVolume/CuLane/ReadyFAT/learn/'
PAth2ValidData=r'/media/vjekod/NewVolume/CuLane/ReadyFAT/valid/'
sufix4data='img/'
sufix4label='label/'
path2list=r'/media/vjekod/NewVolume/CuLane/ReadyFAT/'
filterkeys=['bri','rot','flp','per']


imgslist=glob.glob(Path2LearnData+sufix4data+'*.jpg')
labslist=glob.glob(Path2LearnData+sufix4label+'*.jpg')
imgslist.sort()
labslist.sort()
newsimglist=list()
newslabslist=list()


Valimgslist=glob.glob(PAth2ValidData+sufix4data+'*.jpg')
Vallabslist=glob.glob(PAth2ValidData+sufix4label+'*.jpg')
Valimgslist.sort()
Vallabslist.sort()

#check for pairs :
learn_data_pairs=list()
for img in imgslist:
    for lab in labslist:
        if img.split('/')[-1][0:-10]==lab.split('/')[-1][0:-10]:
            learn_data_pairs.append([img,lab])
            labslist.remove(lab)
            print('learn%=',len(learn_data_pairs)/len(imgslist))
            break


#check for pairs :
valid_data_pairs=list()
for img in Valimgslist:
    for lab in Vallabslist:
        if img.split('/')[-1][0:-10]==lab.split('/')[-1][0:-10]:
            valid_data_pairs.append([img,lab])
            Vallabslist.remove(lab)
            print('valid%=', len(valid_data_pairs) / len(Valimgslist))
            break



import csv

with open(path2list+'LearnFull.csv', 'w') as f:
    # create the csv writer
    writer = csv.writer(f)
    # write a row to the csv file
    for each in learn_data_pairs:
        row=(['/'.join(each[0].split('/')[-3:]),'/'.join(each[1].split('/')[-3:])])
        writer.writerow(row)


with open(path2list+'ValidFull.csv', 'w') as f:
    # create the csv writer
    writer = csv.writer(f)
    # write a row to the csv file
    for each in valid_data_pairs:
        row=(['/'.join(each[0].split('/')[-3:]),'/'.join(each[1].split('/')[-3:])])
        writer.writerow(row)

with open(path2list+'LearnOrigOnly.csv', 'w') as f:
    # create the csv writer
    writer = csv.writer(f)
    # write a row to the csv file
    for each in learn_data_pairs:
        key=((each[0].split('/')[-1])[0:-10]).split('_')[-2]
        if key not in filterkeys:
            print(each[0])
            row=(['/'.join(each[0].split('/')[-3:]),'/'.join(each[1].split('/')[-3:])])
            writer.writerow(row)


print("wait here")