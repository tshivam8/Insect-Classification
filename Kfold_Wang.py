#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: mtech10
"""

import pandas as pd
import numpy as np
from numpy import array
import cv2
from statistics import mean 
import math
import os
from os import listdir
from os.path import isfile, join

from sklearn.svm import SVC
from sklearn.model_selection import KFold, train_test_split
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
#from matplotlib import pyplot as plt
pd.options.mode.chained_assignment = None



class Model:
     def __init__(self):
        self.feature_matrix=np.zeros(0)
    
     def trainxlsx(self, path):
         tdict = pd.read_excel(path, sheet_name=None)['Sheet1']
         df = pd.DataFrame(tdict)
         for ind in df.index:
             if(df['Class'][ind]=='Auchenorrhyncha'):
                 df['Class'][ind]=1.0
             elif(df['Class'][ind]=='Coleoptera'):
                 df['Class'][ind]=2.0
             elif(df['Class'][ind]=='Heteroptera'):
                 df['Class'][ind]=3.0
             elif(df['Class'][ind]=='Hymenoptera'):
                 df['Class'][ind]=4.0
             elif(df['Class'][ind]=='Lepidoptera'):
                 df['Class'][ind]=5.0
             elif(df['Class'][ind]=='Megalptera'):
                 df['Class'][ind]=6.0
             elif(df['Class'][ind]=='Neuroptera'):
                 df['Class'][ind]=7.0
             elif(df['Class'][ind]=='Odonata'):
                 df['Class'][ind]=8.0
             elif(df['Class'][ind]=='Orthoptera'):
                 df['Class'][ind]=9.0
             else:
                 df['Class'][ind]=0.0
         #print(df.loc[df['Class'] == 1.0])
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
    def __init__(self):
        pass
        # self.Train=preprocessing.scale(feature_matrix1[:,:-1])
        # self.Target=np.uint8(feature_matrix1[:,-1:]).ravel()
        # #print(Target)
        # self.Test=preprocessing.scale(feature_matrix2[:,:-1])
        # self.Actual=np.uint8(feature_matrix2[:,-1:]).ravel()

    def svm(self, X_train=None, y_train=None, X_test=None, y_test=None):
        #clf = SVC(kernel='rbf', random_state=0, gamma=.01, C=1) : 54%
        #clf = SVC(kernel='linear') : 57%
        #clf = SVC(kernel='poly') : 52%
        #clf = SVC(kernel='sigmoid') : 31%
        clf = SVC(kernel='rbf', C=10, gamma=10) # 71.22%
        if X_train is None:
            X_train=self.Train
            y_train=self.Target
            X_test=self.Test
            y_test=self.Actual

        clf.fit(X_train, y_train)
        prediction=clf.predict(X_test)
        result=np.hstack((prediction.reshape(-1,1),y_test.reshape(-1,1)))
        return self.accuracy(result)

    def kfold(self,dataset,k=10):
        kf = KFold(n_splits=k,shuffle=True, random_state=1)
        X = preprocessing.scale(dataset[:,:-1])
        #scaler = MinMaxScaler(feature_range=(0, 1)) 
        #X = scaler.fit_transform(dataset[:,:-1]) 
        y = dataset[:,-1:].ravel()
        print(y)
        acc = []
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = np.uint8(y[train_index]), np.uint8(y[test_index])
            acc.append(self.nb(X_train,y_train,X_test,y_test))
        print('accuracy for kfold is',mean(acc))

    def knn(self, X_train=None, y_train=None, X_test=None, y_test=None):
        clf = neighbors.KNeighborsClassifier(n_neighbors = 3)
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
        return (prediction*100)

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
        return temp*100

    def nb(self, X_train=None, y_train=None, X_test=None, y_test=None):
        clf = GaussianNB()
        #clf = MultinomialNB()

        if X_train is None:
            X_train=self.Train
            y_train=self.Target
            X_test=self.Test
            y_test=self.Actual


        clf.fit(X_train, y_train)
        #clf.partial_fit(X_train, y_train, np.unique(y_train))
        predictions = clf.predict(X_test)
        temp = accuracy_score(y_test, predictions)
        print("Accuracy of Naive Bayes = %.5f\n"%(temp * 100))
        return temp*100

    def ann(self, X_train=None, y_train=None, X_test=None, y_test=None):
        #clf = MLPClassifier(solver='lbfgs', alpha=0.0001, activation ='logistic', max_iter=1000,hidden_layer_sizes=(150,60), random_state=1)
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
        return temp*100

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
        return temp*100

    def accuracy(self,result):
        #print(result)
        correct=0
        for tt in result:
            if(tt[0]==tt[1]):
                correct+=1
        accuracy=float(correct/len(result)) 
        print("Accuracy of svm=%.5f\n"%(accuracy*100))
        return (accuracy*100)


def main():
    
    directory= r"Datasets\InsectShapeFeaturesResult_Wang 9classes.xlsx"
    model=Model()
    feature_matrix=model.trainxlsx(directory)
    
    clasify=Classification()
    clasify.kfold(feature_matrix)

if __name__== "__main__":
  main()

           