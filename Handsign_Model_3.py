import cv2
import random
import numpy as np
import cPickle as pickle
from matplotlib import pyplot as plt

svm_params = dict(kernel_type = cv2.SVM_LINEAR,
					svm_type =  cv2.SVM_C_SVC,
					C=2.67, gamma=5.383)

#Preprocessing
def skin_detection(img):
	skinLow = np.array([0,48,80],np.uint8)
	skinHigh = np.array([20,255,255],np.uint8)
    
	imgHSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
	mask = cv2.inRange(imgHSV,skinLow,skinHigh)
	result = cv2.bitwise_and(img,img,mask=mask)
	imgGray = cv2.cvtColor(result,cv2.COLOR_BGR2GRAY)

	return imgGray

#Feature Descriptor
winSize = (128,128)
blockSize = (16,16)
blockStride = (4,4)
cellSize = (8,8)
nbins = 9
derivAperture = 1
winSigma = -1.
histogramNormType = 0
L2HysThreshold = 0.2
gammaCorrection = 1
nlevels = 64

hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,
                        derivAperture,winSigma,histogramNormType,
                        L2HysThreshold,gammaCorrection,nlevels)

    
pre = []
preTest = []
hogData1 = []
hogData2 = []
hogDataTest1 = []
hogDataTest2 = []

img = pickle.load(open("img_save.p","rb"))

imgTrain = (img[4:18]+img[22:36]+img[40:54]+img[58:72]+img[76:90]+
            img[94:108]+img[112:126]+img[130:144]+img[148:162]+img[166:180]+
            img[184:198]+img[202:216]+img[220:234]+img[238:252]+img[256:270]+
            img[274:288]+img[292:306]+img[310:324]+img[328:342]+img[346:360]+
            img[364:378]+img[382:396]+img[400:414]+img[418:432])

imgTest = (img[0:4]+img[18:22]+img[36:40]+img[54:58]+img[72:76]+
            img[90:94]+img[108:112]+img[126:130]+img[144:148]+img[162:166]+
            img[180:184]+img[198:202]+img[216:220]+img[234:238]+img[252:256]+
            img[270:274]+img[288:292]+img[306:310]+img[324:328]+img[342:346]+
            img[360:364]+img[378:382]+img[396:400]+img[414:418])

svm = cv2.SVM()

x = 1
#TrainData
for n in imgTrain:
    print("proses "+str(x))
    x+=1	
    pre.append(skin_detection(n))
for n in pre:
    hogData1.append(hog.compute(n))
for n in hogData1:
    hogData2.append(n.flatten())

TrainData = np.float32(hogData2).reshape(-1,30276)
responses = np.float32(np.repeat(np.arange(24),14)[:,np.newaxis])#model diganti dengan nilai float
svm.train(TrainData,responses,params=svm_params)
svm.save('svm_data.dat')

#TestData
for n in imgTest:
	preTest.append(skin_detection(n))
for n in preTest:
	hogDataTest1.append(hog.compute(n))
for n in hogDataTest1:
    hogDataTest2.append(n.flatten())

TestData = np.float32(hogDataTest2).reshape(-1,30276)
responTest = np.float32(np.repeat(np.arange(24),4)[:,np.newaxis])#model diganti dengan nilai float
result = svm.predict_all(TestData)

#Check Accuracy
mask = result==responTest
correct = np.count_nonzero(mask)
print("nilai akurasi :"+str(correct*100.0/result.size))