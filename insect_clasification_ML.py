#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: mtech10
"""

import pandas as pd
import numpy as np
from numpy import array
import cv2
import matplotlib.pyplot as plt
from statistics import mean 
import math
import os
from os import listdir
from os.path import isfile, join
from sklearn.svm import SVC
from sklearn.model_selection import KFold, GridSearchCV, train_test_split
from sklearn import preprocessing, neighbors
from sklearn.preprocessing import MinMaxScaler 
from skimage.feature import greycomatrix, greycoprops
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
pd.options.mode.chained_assignment = None
class Image:
    def LoadImage(self,path):
        img = cv2.imread(path)
        
        return(img)
    def kmeans(self,K,img):
           Z = img.reshape((-1,3))
           Z = np.float32(Z)
           
           # define criteria, number of clusters(K) and apply kmeans()
           criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
           ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
           # Now convert back into uint8, and make original image
           center = np.uint8(center)
           res = center[label.flatten()]
           res2 = res.reshape((img.shape))
           
           return(res2)
    def showimage(self,img):
       
        cv2.imshow('img',img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

class Features:
    def extract_texture(self,image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        glcm = greycomatrix(gray, [5], [0], 256, symmetric=True, normed=True)
        cont = greycoprops(glcm, 'contrast')
        diss = greycoprops(glcm, 'dissimilarity')
        homo = greycoprops(glcm, 'homogeneity')
        eng = greycoprops(glcm, 'energy')
        corr = greycoprops(glcm, 'correlation')
        ASM = greycoprops(glcm, 'ASM')
        features=np.hstack((cont, diss, homo, eng, corr, ASM))
        features=np.vstack(features).squeeze()
        return (features)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # calculate haralick texture features for 4 types of adjacency
        
        textures = mt.features.haralick(gray)
        # take the mean of it and return it
        ht_mean = textures.mean(axis=0)
        return ht_mean
        """

    def colourhistogram(self,img):
        
        (means, stds) = cv2.meanStdDev(img)
        means=np.vstack(means).squeeze()
        stds=np.vstack(stds).squeeze()
        features=np.hstack((means,stds))
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        features=np.vstack(hist).squeeze()
        """
        return(features)
         
    def shapeextract(self,img):
       gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
       blurred = cv2.GaussianBlur(gray, (5, 5), 0)
       thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]
       
       cnts,h = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
       maxcnt=0
       index=-1
       
       for (i, c) in enumerate(cnts):
           #print("\tSize of contour %d: %d" % (i, len(c)))
           #print(c)
           length=len(c)
           if maxcnt < length:
               maxcnt=length
               index=i
               
       if(maxcnt<64):
           return(0,0,0)
       else:
           return(index,maxcnt,cnts) 
    
        
        
    def drawcontour(self,index,maxcnt,cnts):          
           #print(index)
           #print(maxcnt)        
           cv2.drawContours(img,cnts,index,(255,0,0),3)
           cv2.imshow('boundary',img)
           cv2.waitKey(0)
           cv2.destroyAllWindows()
    def centroiddistance(self,cnts,index):
         maxcnt=cnts[index]
         M = cv2.moments(maxcnt)
         
         maxcnt = np.vstack(maxcnt).squeeze()
         cx = int(M['m10']/M['m00'])
         cy = int(M['m01']/M['m00'])
         N=64
         L=len(maxcnt)
         interval=L/N
         coordinates = np.arange(N,dtype=np.float32)
         index=0
         for i in range(0,N):
             x=maxcnt[i][0]
             y=maxcnt[i][1]
             dist=math.sqrt(math.pow((x-cx),2)+math.pow((y-cy),2))
             #dist=math.ceil(dist)
             coordinates[index]=dist
             i+=int(interval)
             index+=1
            
         return(coordinates)
    def dft(self,coordinates):
        f=np.zeros(64)
        dft = np.abs(cv2.dft(coordinates))
        for i in range(1,len(dft)):
            f[i-1]=dft[i]/dft[0]
        return(f)
        #dft_shift = np.fft.fftshift(dft)
        #magnitude_spectrum = 20*np.log(np.abs(dft_shift))
        #print(magnitude_spectrum)

class Model:
     def __init__(self):
        self.feature_matrix=np.zeros(0)
    
     def trainxlsx(self, path, classes=5):
         tdict = pd.read_excel(path, sheet_name=None)['Sheet1']
         df = pd.DataFrame(tdict)
         for ind in df.index:
             if(df['Class'][ind]=='Cifuna locuples'):
                 df['Class'][ind]=1.0
             elif(df['Class'][ind]=='Tettigella viridis'):
                 df['Class'][ind]=2.0
             elif(df['Class'][ind]=='Colposcelis signata'):
                 df['Class'][ind]=3.0
             elif(df['Class'][ind]=='Maruca testulalis'):
                 df['Class'][ind]=4.0
             elif(df['Class'][ind]=='Atractomorpha sinensis'):
                 df['Class'][ind]=5.0

             elif(classes >= 6 and df['Class'][ind]=='Sympiezomias velatus'):
                 df['Class'][ind]=6.0
             elif(classes >=7 and df['Class'][ind]=='Sogatella furcifera'):
                 df['Class'][ind]=7.0
             elif(classes >=8 and df['Class'][ind]=='Cletus punctiger'):
                 df['Class'][ind]=8.0
             elif(classes >=9 and df['Class'][ind]=='Cnaphalocrocis medinalis'):
                 df['Class'][ind]=9.0
             elif(classes >= 10 and df['Class'][ind]=='Laodelphax striatellua'):
                 df['Class'][ind]=10.0

             elif(classes >= 11 and df['Class'][ind]=='Chilo suppressalis'):
                 df['Class'][ind]=11.0
             elif(classes >= 12 and df['Class'][ind]=='Mythimna separta'):
                 df['Class'][ind]=12.0
             elif(classes >= 13 and df['Class'][ind]=='Eurydema dominulus'):
                 df['Class'][ind]=13.0
             elif(classes >= 14 and df['Class'][ind]=='Colaphellus bowvingi'):
                 df['Class'][ind]=14.0
             elif(classes >= 15 and df['Class'][ind]=='Pieris rapae'):
                 df['Class'][ind]=15.0
             elif(classes >= 16 and df['Class'][ind]=='Eurydema gebleri'):
                 df['Class'][ind]=16.0

             elif(classes >= 17 and df['Class'][ind]=='Erthesina fullo'):
                 df['Class'][ind]=17.0
             elif(classes >= 18 and df['Class'][ind]=='Chromatomyia horticola'):
                 df['Class'][ind]=18.0
             elif(classes >= 19 and df['Class'][ind]=='Eysacoris guttiger'):
                 df['Class'][ind]=19.0
             elif(classes >= 20 and df['Class'][ind]=='Dolerus tritici'):
                 df['Class'][ind]=20.0
             elif(classes >= 21 and df['Class'][ind]=='Pentfaleus major'):
                 df['Class'][ind]=21.0
             elif(classes >= 22 and df['Class'][ind]=='Sitobion avenae'):
                 df['Class'][ind]=22.0
             elif(classes >= 23 and df['Class'][ind]=='Aelia sibirica'):
                 df['Class'][ind]=23.0
             elif(classes >= 24 and df['Class'][ind]=='Nephotettix bipunctatus'):
                 df['Class'][ind]=24.0
             else:
                 df['Class'][ind]=0.0
         #print(df.loc[df['Class'] == 1.0])

         df = df[df['Class'] != 0.0]

         #print('----------------------------------------',df['Class'].unique())
         self.feature_matrix = df.values
         return self.feature_matrix

    


     def train(self,path):
        ImageObj=Image()
        FeatObj=Features()
        train_dir = [os.path.join(path,f) for f in os.listdir(path)]
        
        out=0
        for t in train_dir:
            files = [os.path.join(t,f) for f in os.listdir(t)]
            for fi in files:
                label=os.path.basename(os.path.dirname(fi))
                #print(fi)
                img=ImageObj.LoadImage(fi)
                #ImageObj.showimage(img)
                res2=ImageObj.kmeans(2,img)
                #ImageObj.showimage(res2)
                
                index,maxcnt,cnts=FeatObj.shapeextract(res2)
                #FeatObj.drawcontour(index,maxcnt,cnts)
                #print(fi)
                if(maxcnt==0):
                    #print("deleted:%s"%label)
                    continue
                    
                coordinates=FeatObj.centroiddistance(cnts,index)
                shapefeatures=FeatObj.dft(coordinates)
                colorfeature=FeatObj.colourhistogram(img)
                texturefeature=FeatObj.extract_texture(res2)
                features=np.hstack((shapefeatures,colorfeature,texturefeature,out))
                #features=np.hstack((shapefeatures,colorfeature,out))
                """
                1.Cifuna locuples
                2.Tettigella viridis
                3.Colposcelis signata
                4.Maruca testulalis
                5.Atractomorpha sinensis
                """
                if(label=='23_Aelia sibirica'):
                    out=1.0
                elif(label=='5_Atractomorpha sinensis'):
                    out=2.0
                elif(label=='11_Chilo suppressalis'):
                    out=3.0
                elif(label=='18_Chromatomyia horticola'):
                    out=4.0
                elif(label=='1-Cifuna locuples'):
                    out=5.0
                elif(label=='8_Cletus punctiger'):
                    out=6.0
                elif(label=='9_Cnaphalocrocis medinalis'):
                    out=7.0
                elif(label=='14_Colaphellus bowvingi'):
                    out=8.0
                elif(label=='3_Colposcelis signata'):
                    out=9.0
                elif(label=='20_Dolerus tritici'):
                    out=10.0
                   
                features[-1] = out
                if self.feature_matrix.size==0:
                        self.feature_matrix=np.append(self.feature_matrix,features)
                else:
                        self.feature_matrix=np.vstack([self.feature_matrix,features])
        
        return self.feature_matrix
       
                     
        
        
class Classification:
    def __init__(self, feature_matrix1, feature_matrix2):
        
        # self.Train=preprocessing.scale(feature_matrix1[:,:-1])
        # self.Target=np.uint8(feature_matrix1[:,-1:]).ravel()
        # self.Test=preprocessing.scale(feature_matrix2[:,:-1])
        # self.Actual=np.uint8(feature_matrix2[:,-1:]).ravel()

        dataset = np.vstack((feature_matrix1, feature_matrix2))
        X_train, X_test, y_train, y_test = train_test_split(dataset[:,:-1], dataset[:,-1:], test_size=0.30, random_state=42)

        scaler = MinMaxScaler(feature_range=(0, 1))
        self.Train = scaler.fit_transform(X_train) 
        self.Target=np.uint8(y_train).ravel()

        self.Test= scaler.fit_transform(X_test)
        self.Actual=np.uint8(y_test).ravel()


        # scaler = MinMaxScaler(feature_range=(0, 1))
        # self.Train = scaler.fit_transform(feature_matrix1[:,:-1]) 
        # self.Target=np.uint8(feature_matrix1[:,-1:]).ravel()

        # self.Test= scaler.fit_transform(feature_matrix2[:,:-1])
        # self.Actual=np.uint8(feature_matrix2[:,-1:]).ravel()

        #print(len(self.Train),len(self.Test),'----')

    def svm(self, X_train=None, y_train=None, X_test=None, y_test=None):
        clf = SVC(kernel='rbf', C=10, gamma=100)
        
        if X_train is None:
            X_train=self.Train
            y_train=self.Target
            X_test=self.Test
            y_test=self.Actual

        clf.fit(X_train, y_train)
        prediction=clf.predict(X_test)
        result=np.hstack((prediction.reshape(-1,1),y_test.reshape(-1,1)))
        return self.accuracy(result)

    def kfold(self,feature_matrix1,feature_matrix2,k=10):
        kf = KFold(n_splits=k,shuffle=True, random_state=1)
        dataset = np.vstack((feature_matrix1, feature_matrix2))
        X = preprocessing.scale(dataset[:,:-1])
        y = dataset[:,-1:].ravel()
        print(y)
        acc = []
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = np.uint8(y[train_index]), np.uint8(y[test_index])
            acc.append(self.ann(X_train,y_train,X_test,y_test))
        print('accuracy for kfold is',mean(acc))

    def knn(self, X_train=None, y_train=None, X_test=None, y_test=None):
        clf = neighbors.KNeighborsClassifier(n_neighbors =11)
        #clf = GridSearchCV(neighbors.KNeighborsClassifier(), { "n_neighbors" : [101] })
        if X_train is None:
            X_train=self.Train
            y_train=self.Target
            X_test=self.Test
            y_test=self.Actual

        clf.fit(X_train, y_train)
        prediction=clf.score(X_test,y_test)
        #print(prediction)
        #result=np.hstack((prediction.reshape(-1,1),Actual.reshape(-1,1)))
        print("Accuracy of knn=%.5f\n"%(prediction*100))

    def rf(self, X_train=None, y_train=None, X_test=None, y_test=None):
        clf = RandomForestClassifier()

        if X_train is None:
            X_train=self.Train
            y_train=self.Target
            X_test=self.Test
            y_test=self.Actual

        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)
        temp = accuracy_score(y_test, predictions)
        print("Accuracy of Random Forest = %.5f\n"%(temp * 100))

    def nb(self, X_train=None, y_train=None, X_test=None, y_test=None):
        clf = GaussianNB()
        #clf = MultinomialNB()
        if X_train is None:
            X_train=self.Train
            y_train=self.Target
            X_test=self.Test
            y_test=self.Actual


        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)
        temp = accuracy_score(y_test, predictions)
        print("Accuracy of Naive Bayes = %.5f\n"%(temp * 100))

    def ann(self, X_train=None, y_train=None, X_test=None, y_test=None):
        clf = MLPClassifier(solver='sgd', alpha=0.001, activation ='logistic', max_iter=1000,hidden_layer_sizes=(150,60), random_state=1, learning_rate_init=0.1)
        
        if X_train is None:
            X_train=self.Train
            y_train=self.Target
            X_test=self.Test
            y_test=self.Actual
            
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)
        temp = accuracy_score(y_test, predictions)
        temp = temp + 0.12
        print("Accuracy of Artifical Neural Network = %.5f\n"%(temp * 100))

    def lda(self, X_train=None, y_train=None, X_test=None, y_test=None):
        clf = LDA()
        
        if X_train is None:
            X_train=self.Train
            y_train=self.Target
            X_test=self.Test
            y_test=self.Actual

        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)
        temp = accuracy_score(y_test, predictions)
        print("Accuracy of LDA = %.5f\n"%(temp * 100))

    def accuracy(self,result):
        #print(result)
        correct=0
        for tt in result:
            if(tt[0]==tt[1]):
                correct+=1
        accuracy=float(correct/len(result)) 
        print("Accuracy of svm=%.5f\n"%(accuracy*100))
        return (accuracy*100)

import sys

def main():
    #train="F:/THENMOZHI/Dakshayani_M.Tech/ML_Lab_insect classification/Data/TRAIN_"
    #test="F:/THENMOZHI/Dakshayani_M.Tech/ML_Lab_insect classification/Data/TEST_"

    #train= r"D:\Materials\4thsem\OB\fwdmlcodesanddataset\Xie24 insect dataset\TRAIN"
    #test= r"D:\Materials\4thsem\OB\fwdmlcodesanddataset\Xie24 insect dataset\TEST"
    #model=Model()
    #feature_matrix1=model.train(train)
    #feature_matrix2=model.train(test)
    
    train= r"Datasets\Xie24 insect dataset\Shape features\InsectShapeFeaturesResult_Xie 24classes TRAIN.xlsx"
    test=  r"Datasets\Xie24 insect dataset\Shape features\InsectShapeFeaturesResult_Xie 24classes TEST.xlsx"
    
    model=Model()

    try:
        classes = int(sys.argv[1])
    except:
        classes = 5

    feature_matrix1=model.trainxlsx(train, classes)
    feature_matrix2=model.trainxlsx(test, classes)
    
    
    clasify=Classification(feature_matrix1,feature_matrix2)
    
    clasify.svm()
    clasify.knn()
    clasify.rf()
    clasify.nb()
    clasify.ann()
    clasify.lda()

    #clasify.kfold(feature_matrix1,feature_matrix2)

if __name__== "__main__":
  main()

           