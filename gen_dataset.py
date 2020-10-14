#!/usr/bin/env python
# coding: utf-8

# Generating data set + splitting functions


import numpy as np
import random
import math

def make_samples(no_of_samples=100):
        
    images = np.full((no_of_samples,8,8),255)
    
    choices = [0,1]                             #Left or Right (OR) Top or bottom 
    r_c = [1,2,3,4,5,6,7]                       #Width of darker region
    labels = []
    
    for i in range(no_of_samples):
        l = random.choice(r_c)
        
        if i<(no_of_samples//2):
            labels.append([0])
            if random.choice(choices):
                images[i,:,:l] = np.random.randint(low=0,high=129,size=(8,l))
            else:
                images[i,:,l:] = np.random.randint(low=0,high=129,size=(8,8-l))
        else:
            labels.append([1])
            if random.choice(choices):
                images[i,:l,:] = np.random.randint(low=0,high=129,size=(l,8))
            else:
                images[i,l:,:] = np.random.randint(low=0,high=129,size=(8-l,8))
                

    return images, (np.array(labels,dtype=np.int32).T)

def split(X,Y):
    '''
    Splits data and shuffles together
    '''
    
    X_train = np.zeros((700,8,8))
    Y_train =np.zeros((2,700))
    X_test = np.zeros((300,8,8))
    Y_test =np.zeros((2,300))
    
    #Train set to have equal no. of each class (700 examples)
    X_train[:350] = X[:350]
    X_train[350:] = X[500:850]
    Y_train[:,:350] = Y[:,:350]
    Y_train[:,350:] = Y[:,500:850]
    
    #Test set to have equal no. of each class (300 examples)
    X_test[:150] = X[350:500]
    X_test[150:] = X[350:500]
    Y_test[:,:150] = Y[:,850:]
    Y_test[:,150:] = Y[:,850:]    
    
    #Shuffle each
    permutation1 = np.random.permutation(len(X_train))
    X_train = X_train[permutation1].reshape((700,8,8,1))
    Y_train = Y_train[:,permutation1]
    
    permutation2 = np.random.permutation(len(X_test))
    X_test = X_test[permutation2].reshape((300,8,8,1))
    Y_test = Y_test[:,permutation2]
    
    
    return X_train, Y_train, X_test, Y_test

