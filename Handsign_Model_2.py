import cv2
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

imgTrain = (img[6:18]+img[24:36]+img[42:54]+img[60:72]+img[78:90]+
            img[96:108]+img[114:126]+img[132:144]+img[150:162]+img[168:180]+
            img[186:198]+img[204:216]+img[222:234]+img[240:252]+img[258:270]+
            img[276:288]+img[294:306]+img[312:324]+img[330:342]+img[348:360]+
            img[366:378]+img[384:396]+img[402:414]+img[420:432])

imgTest = (img[0:6]+img[18:24]+img[36:42]+img[54:60]+img[72:78]+
            img[90:96]+img[108:114]+img[126:132]+img[144:150]+img[162:168]+
            img[180:186]+img[198:204]+img[216:222]+img[234:240]+img[252:258]+
            img[270:276]+img[288:294]+img[306:312]+img[324:330]+img[342:348]+
            img[360:366]+img[378:384]+img[396:402]+img[414:420])

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
responses = np.float32(np.repeat(np.arange(24),12)[:,np.newaxis])#model diganti dengan nilai float
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
responTest = np.float32(np.repeat(np.arange(24),6)[:,np.newaxis])#model diganti dengan nilai float
result = svm.predict_all(TestData)

#Check Accuracy
mask = result==responTest
correct = np.count_nonzero(mask)
print("nilai akurasi :"+str(correct*100.0/result.size))