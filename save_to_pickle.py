import cPickle as pickle
import os
import cv2

path = os.getcwd()+"/Dataset_II"
dataset = os.listdir(path)
dataset = sorted(dataset)
label = [i.split('_')[0] for i in dataset]

ori = []
img = []

for n in dataset:
    print("membaca gambar : "+n)
    ori.append(cv2.imread(path+"/"+n))
for n in ori:     
    img.append(cv2.resize(n,(128,128)))

pickle.dump(img,open("imgII_save.p","wb"))
