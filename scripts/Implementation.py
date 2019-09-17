# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 15:57:52 2019

@ COMP 551: Applied Machine Learning (Winter 2019) 
@ Mini-project 1 : Implementing Logistic Regression and LDA from scratch 

# Team Members: 

@ Hair Albeiro Parra Barrera 
@ ID: 260738619 

@ Sun Gengyi 
@ ID: 

"""
# *****************************************************************************

### *** Implementing Logistic Regression and LDA from scratch *** ### 

# ******************************************************************************

### 1. Imports ### 

import scipy
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from tqdm import tqdm # to display progress bar
from numpy import transpose as T # because it is a pain in the ass

# *****************************************************************************

### 2. Loading the Data, Preprocessing  & Statistics ### 

## 2.1 Load the red wine dataset
red_wine_df = pd.read_csv('./Datasets/winequality-red.csv', sep = ';') 

# Convert multi-target problem into a binary problem by aassigning only 2 classes: 
red_wine_df.loc[red_wine_df['quality'] > 5, 'quality'] = 'Good'  # Good quality
red_wine_df.loc[red_wine_df['quality'] != 'Good', 'quality'] = 'Bad' # Bad quality 

# note that the data is all in different scales, therefore we will normalize 
# each feature column/ 

def normalize_df(df, columns=[]): 
    """ 
    Normalizes specifies columns of the input dataframe. 
    Parameters: 
        @ df: the input dataframe
        @ columns: a list of strings, which are the name of the
                   columns to normalize. 
    """
    for column in columns: 
        col_avg = np.mean(df[column]) # get average 
        col_std = np.std(df[column]) 
        df[column] = df[column].apply(lambda x: ((x-col_avg) / col_std) ) 
        

# Get column names 
red_wine_df_cols = list(red_wine_df.columns)[:-1] # don't include the targets

# Pairplot without normalization 
sns.pairplot(red_wine_df.drop('quality', axis= 1))

# Normalize
normalize_df(red_wine_df, columns=red_wine_df_cols)

# Statistics
print("Red wine df shape: {}".format(red_wine_df.shape)) 
print("Red wine 'Good' counts: ", red_wine_df['quality'][red_wine_df['quality'] == 'Good'].count() )
print("Red wine 'Bad' counds: ", red_wine_df['quality'][red_wine_df['quality'] == 'Bad'].count() )
red_wine_df_stats = red_wine_df.drop('quality', axis=1).describe()

# Label distribution plot
sns.countplot(red_wine_df['quality'])

# Features pairplot
sns.pairplot(red_wine_df.drop('quality', axis= 1), diag_kind='kde')


## 2.2 Load the breast cancer dataset 


# The attributes are given the following way: 
#   #  Attribute                     Domain
#   -- -----------------------------------------
#   1. Sample code number            id number
#   2. Clump Thickness               1 - 10
#   3. Uniformity of Cell Size       1 - 10
#   4. Uniformity of Cell Shape      1 - 10
#   5. Marginal Adhesion             1 - 10
#   6. Single Epithelial Cell Size   1 - 10
#   7. Bare Nuclei                   1 - 10
#   8. Bland Chromatin               1 - 10
#   9. Normal Nucleoli               1 - 10
#  10. Mitoses                       1 - 10
#  11. Class:                        (2 for benign, 4 for malignant)

# Declare column names 
cancer_cols= ['id',
           'Clump Thickness',
           'Uniformity of Cell Size',
           'Uniformity of Cell Shape',
           'Marginal Adhesion', 
           'Single Epithelial Cell Size',
           'Bare Nuclei',
           'Bland Chromatin', 
           'Normal Nucleoli',
           'Mitoses',
           'Class']

# Data preparation and cleaning
cancer_df =  pd.read_csv('./Datasets/breast-cancer-wisconsin.data') # read data
cancer_df.columns = cancer_cols  # rename columns 
cancer_df = cancer_df.dropna().drop('id', axis=1) # drop na rows and id
cancer_df = cancer_df[cancer_df['Bare Nuclei'] != '?'] # drop rows with missing vals 
cancer_df['Bare Nuclei'] = pd.to_numeric(cancer_df['Bare Nuclei']) # change back to int
cancer_df.loc[cancer_df['Class'] == 4, 'Class' ] = 'malignant' # malignant
cancer_df.loc[cancer_df['Class'] == 2, 'Class' ] = 'benign' # benign

# Fix data type 
for column in cancer_df.columns[:-1]: 
    print(column)
    cancer_df[column] = pd.to_numeric(cancer_df[column])

# Print shape and obtain statistics. 
print("cancer_df shape: {}".format(cancer_df.shape)) 
print("cancer data 'Benign' counts: ", cancer_df['Class'][cancer_df['Class'] == 'benign'].count() )
print("cancer data 'Malignant' counts: ", cancer_df['Class'][cancer_df['Class'] == 'malignant'].count() )
cancer_df_stats = cancer_df.drop('Class', axis=1).describe()

# Labels distribution plot 
sns.countplot(cancer_df['Class'])

# Features pairplot
sns.pairplot(cancer_df.drop('Class',axis=1), diag_kind='kde')

# Since the all the features seem to lie in similar ranges, we will not
# normalize them. 


# 2.2 Separate into Features and labels for both  

# Red wine data features and labels
X_redwine = red_wine_df.drop('quality', axis=1).values 
y_redwine = red_wine_df['quality']

# Cancer data features and labels 
X_cancer = cancer_df.drop('Class', axis=1).values.astype('float64')
y_cancer = cancer_df['Class']


# *****************************************************************************

### 3. Helper Functions ### 


# *****************************************************************************

### 4. Implementing Logistic Regression ### 

class LogisticRegression(): 
    
    np.random.seed(42) 
    
    def __init__(self, X, y): 
        """
        X: (n x m) matrix of features with n observations 
         and m features. (in np matrix format)
        y: (n x 1) vector of targets (as Series)
        """ 
        # Verify dimensions
        if X.shape[0] != y.shape[0]: 
            message = "Input dimensions don't match" 
            message += "\n X is {} but y is {}".format(X.shape, y.shape)
            raise ValueError(message) 
            
        self.n = X.shape[0]
        self.m = X.shape[1] + 1 # Because of intercept term 
        X_0 = np.ones(self.n) # intercept features 
        self.X = np.c_[X_0,X] # concatenate 
        self.w = np.random.rand(self.m, 1) # initialize weights 
        self.y = y 
        
        print("Initialized with dimensions X.shape=({}) y.shape=({})".format(self.X.shape, self.y.shape)) 
        print("Number of features: m={}".format(self.m)) 
        print("Number of observations: n={}".format(self.n)) 
        print("Number of weights: len(w)={}".format(len(self.w)))
        
        
    def sigmoid(self, x): 
        return 1/(1+ np.exp(-x))
    
    def d_sigmoid(self, x): 
        return self.sigmoid(x)*(1-self.sigmoid(x)) 

    


    


obj = LogisticRegression(X_redwine, y_redwine)
obj.sigmoid(4)


class LinearRegression(): 
    
    np.random.seed(42)
    
    def __init__(self, X, y): 
        """
        X: (n x m) matrix of features with n observations 
         and m features. 
        y: (n x 1) vector of targets
        """        
        
        if X.shape[0] != y.shape[0]: 
            message = "Input dimensions don't match" 
            message += "\n X is {} but y is {}".format(X.shape, y.shape)
            raise ValueError(message)
        
        self.n = X.shape[0] 
        self.m = X.shape[1] + 1
        X_0 = np.ones(self. n) # intercept
        self.X = np.c_[X_0, X] # concatenate
        self.w = np.random.rand(self.m, 1) # random initialization 
        self.y = y
        
        print("X: \n", self.X) 
        print("w: \n", self.w) 
        print("y: \n", self.y)
        print("n: ", self.n)
        print("m: ", self.m)
    
    def predict(self, X_new): 
        return np.matmul(X_new, self.w) 
         
    def MSE(self): 
        # (y- Xw)T (y-Xw)
        pred = self.predict(self.X) 
        diff = self.y - pred 
        return np.matmul(T(diff), diff)[0][0]
    
    
    def gradient(self): 
        pred = self.predict(self.X) # Xw
        diff = self.y - pred  # y-Xw
        return -2 * np.matmul(T(self.X), diff) # -2X^T (y-Xw)
    
    
    def train(self, alpha = 0.02, threshold = 0.5, epochs = 10): 
        """
        Trains itself using gradient descent
        """
        prev_error = self.MSE() + 1000 # Initialize error
        
        for k in tqdm(range(epochs), desc="\nTraining...") : 
            
            grad = alpha*self.gradient()            
            temp = self.w - grad
            self.w = temp
            error = self.MSE()
            
            print("\nEpoch {}".format(k+1))
            print("MSE: %.2f" % (error) )
                        
            if abs(error-prev_error) < threshold: 
                break
            
            prev_error = error






















