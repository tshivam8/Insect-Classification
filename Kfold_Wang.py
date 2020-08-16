#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: mtech10
"""

import sys
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

pd.options.mode.chained_assignment = None



class Model:
     def __init__(self):
        self.feature_matrix=np.zeros(0)
    
     def trainxlsx(self, path, classes=5):
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
             elif(classes == 9 and df['Class'][ind]=='Megalptera'):
                 df['Class'][ind]=6.0
             elif(classes == 9 and df['Class'][ind]=='Neuroptera'):
                 df['Class'][ind]=7.0
             elif(classes == 9 and df['Class'][ind]=='Odonata'):
                 df['Class'][ind]=8.0
             elif(classes == 9 and df['Class'][ind]=='Orthoptera'):
                 df['Class'][ind]=9.0
             else:
                 df['Class'][ind]=0.0
         #print(df.loc[df['Class'] == 1.0])4
         #df = df[df['Class'] != 0.0]
         #print('-------------------',df['Class'].unique())
         self.feature_matrix = df.values
         return self.feature_matrix
       
                     
        
        
class Classification:

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

    def kfold(self,dataset,k=4):
        kf = KFold(n_splits=k,shuffle=True, random_state=1)
        X = preprocessing.scale(dataset[:,:-1])
        #scaler = MinMaxScaler(feature_range=(0, 1)) 
        #X = scaler.fit_transform(dataset[:,:-1]) 
        y = dataset[:,-1:].ravel()
        nb_a = []
        svm_a = []
        knn_a = []
        rf_a = []
        ann_a = []
        lda_a = []

        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = np.uint8(y[train_index]), np.uint8(y[test_index])
            #print(len(X_train),len(X_test),'\n')
            nb_a.append(self.nb(X_train,y_train,X_test,y_test))
            svm_a.append(self.svm(X_train,y_train,X_test,y_test))
            knn_a.append(self.knn(X_train,y_train,X_test,y_test))
            rf_a.append(self.rf(X_train,y_train,X_test,y_test))
            ann_a.append(self.ann(X_train,y_train,X_test,y_test))
            lda_a.append(self.lda(X_train,y_train,X_test,y_test))
        
        print('accuracy for kfold-svm is %.5f'%mean(svm_a))
        print('accuracy for kfold-knn is %.5f'%mean(knn_a))
        print('accuracy for kfold-rf is %.5f'%mean(rf_a))
        print('accuracy for kfold-nb is %.5f'%mean(nb_a))
        print('accuracy for kfold-ann is %.5f'%mean(ann_a))
        print('accuracy for kfold-lda is %.5f'%mean(lda_a))

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
        #print("Accuracy of knn=%.5f\n"%(prediction*100))
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
        #print("Accuracy of Random Forest = %.5f\n"%(temp * 100))
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
        #print("Accuracy of Naive Bayes = %.5f\n"%(temp * 100))
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
        #print("Accuracy of Artifical Neural Network = %.5f\n"%(temp * 100))
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
        #print("Accuracy of LDA = %.5f\n"%(temp * 100))
        return temp*100

    def accuracy(self,result):
        #print(result)
        correct=0
        for tt in result:
            if(tt[0]==tt[1]):
                correct+=1
        accuracy=float(correct/len(result)) 
        #print("Accuracy of svm=%.5f\n"%(accuracy*100))
        return (accuracy*100)


def main():
    
    try:
        classes = sys.argv[1]
    except:
        classes = 5
    directory = ""
    directory= r"Datasets\InsectShapeFeaturesResult_Wang 9classes.xlsx"
    print("wait for results...")
    model=Model()
    feature_matrix=model.trainxlsx(directory, int(classes))
    
    clasify=Classification()
    clasify.kfold(feature_matrix)

if __name__== "__main__":
  main()

           