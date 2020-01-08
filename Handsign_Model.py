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
	img = cv2.GaussianBlur(img,(5,5),0)
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

img = pickle.load(open("imgII_save.p","rb"))

imgTrain = (img[4:19]+img[23:38]+img[42:57]+img[61:76]+img[80:95]+
            img[99:114]+img[118:133]+img[137:152]+img[156:171]+img[175:190]+
            img[194:209]+img[213:228]+img[232:247]+img[251:266]+img[270:285]+
            img[289:304]+img[308:323]+img[327:342]+img[346:361]+img[365:380]+
            img[384:399]+img[403:418]+img[422:437]+img[441:456])

imgTest = (img[0:4]+img[19:23]+img[38:42]+img[57:61]+img[76:80]+
            img[95:99]+img[114:118]+img[133:137]+img[152:156]+img[171:175]+
            img[190:194]+img[209:213]+img[228:232]+img[247:251]+img[266:270]+
            img[285:289]+img[204:208]+img[323:327]+img[342:346]+img[361:365]+
            img[380:384]+img[399:403]+img[418:422]+img[437:441])

svm = cv2.SVM()

x = 0
print("Sedang dalam proses Training")
#TrainData
for n in imgTrain:	
    pre.append(skin_detection(n))
for n in pre:
    hogData1.append(hog.compute(n))
for n in hogData1:
    x+=1
    hogData2.append(n.flatten())

TrainData = np.float32(hogData2).reshape(-1,30276)
responses = np.float32(np.repeat(np.arange(24),15)[:,np.newaxis])#model diganti dengan nilai float
svm.train(TrainData,responses,params=svm_params)
svm.save('svm_data.dat')
print("Data training sebanyak : "+str(x)+" data citra")

x = 0
print("Sedang dalam proses Testing")
#TestData
for n in imgTest:
	preTest.append(skin_detection(n))
for n in preTest:
	hogDataTest1.append(hog.compute(n))
for n in hogDataTest1:
    x+=1
    hogDataTest2.append(n.flatten())
print("Data testing sebanyak : "+str(x)+" data citra")

TestData = np.float32(hogDataTest2).reshape(-1,30276)
responTest = np.float32(np.repeat(np.arange(24),4)[:,np.newaxis])#model diganti dengan nilai float
result = svm.predict_all(TestData)

#Check Accuracy
mask = result==responTest
correct = np.count_nonzero(mask)
print("nilai akurasi :"+str(correct*100.0/result.size))
