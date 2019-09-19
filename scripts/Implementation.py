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
    
@ Name 3 
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
red_wine_df.loc[red_wine_df['quality'] > 5, 'quality'] = 1  # Good quality
red_wine_df.loc[red_wine_df['quality'] != 1, 'quality'] = 0 # Bad quality 

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
sns.pairplot(red_wine_df.drop('quality', axis= 1), diag_kind="kde" )

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
cancer_df.loc[cancer_df['Class'] == 4, 'Class' ] = 0 # malignant
cancer_df.loc[cancer_df['Class'] == 2, 'Class' ] = 1 # benign

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
X_redwine = red_wine_df.drop('quality', axis=1).values.astype('float64')
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
    
    def __init__(self,X=np.array([[0]]), y=np.array([0]) ): 
        """
        X: (n x m) matrix of features with n observations 
         and m features. (in np matrix format)
        y: (n x 1) vector of targets (as Series)
        
        """     
        if X[0,0] == 0 and y[0] == 0: 
            print("Default initialization") 
            
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
        
        
    def sigmoid(self, z): 
        return 1/(1+ np.exp(-z))
    
    def predict_probabilities(self, X_new): 
        """
        Returns a probablistic prediction using the model 
        parameters. 
        inputs: 
            @ self
            @ X_new : (n' x m) input vector in list or 
                      numpy format. 
        """
        X_new = np.array(X_new)
        input_shape = X_new.shape
        
        # If input is a vector 
        if len(X_new.shape) == 1: 
            # If input length doesn't match  
            if(len(X_new) + 1 != self.m): 
                message = "Input number of features doesn't match model number of parameters" 
                message += "\nInput is has {} features but model has {} features".format(len(X_new), self.m - 1)
                raise Exception(message)
            else: 
                x = np.insert(X_new, 0, 1) # for clarity 
                wTx = float(np.matmul(T(self.w), x)) 
                sigm_wTx = self.sigmoid(wTx) 
                return [sigm_wTx] 
                
        # if input is a matrix of new examples
        elif input_shape[0] > 1 and input_shape[1] > 1: 
            print("matrix")
#           # if number of attributes don't match 
            if (X_new.shape[1] + 1) != self.m: 
                message = "Input dimensions don't match" 
                message += "\nInput matrix contains {} features, but the model has {} fitted features".format(self.m - 1)
                raise Exception(message)
            # right dimensions
            else: 
                pred_probs = np.zeros((X_new.shape[0],1)) # to store the probs
                X_0 = np.ones((X_new.shape[0],1)) # n-dim vector of ones
                X_new = np.c_[X_0,X_new] # concatenate
                
                # since X_new is a matrix, we have to loop 
                # over each of its rows, which comes out as
                # a column vector 
                for i in range(len(X_new)): 
                    x = X_new[i] # row = example
                    wTx = float(np.matmul(T(self.w),x)) # w^Tx 
                    sigm_wTx = self.sigmoid(wTx)
                    pred_probs[i] = sigm_wTx
                    
                return pred_probs
            
            
    def predict(self, X_new): 
        """
        Returns an array of predictions for the 
        new input. 
        """
        # get predictions 
        probs = self.predict_probabilities(X_new) 
        
        # Use decision boundary
        return [1 if prob >= 0.5 else 0 for prob in probs]
                
    # loss function
    def cross_entropy_loss(self, verbose=False): 
        
        total_loss = []
        # for each datapoint
        for i in range(self.n): 
            x_i = self.X[i] 
            y_i = self.y[i] 
            wTx = float(np.matmul(T(self.w), x_i)) #w^Tx
            sigm_wTx = self.sigmoid(wTx)
            if y_i == 1:
                total_loss.append(-np.log(sigm_wTx)) 
            else: 
                total_loss.append(-np.log(1-sigm_wTx))
            
        if verbose: 
            print("Loss array: \n") 
            print(total_loss)
        
        return np.sum(np.array(total_loss))
    
    def gradient(self):
        """ 
        Calculates the gradient for the Logistic 
        Regression model 
        """
        grad = np.zeros((self.m,)) # initialize gradient
        
        # calcualte gradient of each example 
        # and add together
        for i in range(self.n): 
            x_i = self.X[i] 
            y_i = self.y[i] 
            wTx = float(np.matmul(T(self.w),x_i)) # w^T x
            sigm_wTx = self.sigmoid(wTx)
            grad += x_i*(y_i - sigm_wTx) # add to previous grad
            
        return grad.reshape((len(grad),1))
    
    def train(self, alpha=0.002, threshold=0.01, epochs=100, auto_alpha=1.0): 
        
        # initialize error
        prev_error = self.cross_entropy_loss()
        print("Initial loss: " , prev_error)
        
        for k in tqdm(range(epochs), desc="\nTraining...") : 
            
            grad = alpha*self.gradient() # calculate gradient   
            temp = np.add(self.w, grad) # get weights update
            self.w = temp # update weights
            error = self.cross_entropy_loss() # calculate current error
            
            print("\nEpoch {}".format(k+1))
            print("Cross-entropy loss: %.2f" % (error) )
                        
            if abs(error-prev_error) < threshold: 
                break
            
            prev_error = error 
            alpha = auto_alpha*alpha
            
        print("---Terminated---")        
        
    
    def fit(self, X,y,alpha=0.02, threshold=0.0001, epochs=20): 
        """
        Initializes the model with the input parameters 
        and trains
        """
        self.__init__(X,y) # Initialize with input 
        self.train(self, alpha, threshold, epochs)
        
        
        
    

# TESTS 
obj = LogisticRegression()
obj = LogisticRegression(X_redwine, y_redwine)
obj.sigmoid(4)
obj.cross_entropy_loss()
obj.gradient()

obj.train(epochs = 1000)

obj.fit(X_redwine,y_redwine)

# new vector(test) 
X_new = X_redwine[1:4,:]
X_new.shape
X_new = X_new[0]
X_new = X_new[0:1,:]
shape = X_new.shape

for row in X_new: 
    print(row.shape)

obj.predict_probabilities(X_new)
obj.predict(X_new)

obj.gradient()


# ***********************************************************

### MODEL : WILL NOT EXACTLY USE THIS IN THE ASSG ### 

# ***********************************************************

### 5. Implementing Linear Discriminant Analysis 

## ---> PLEASE IMPLEMENT THE MODEL HERE <--- ### 


# *********************************************************




# *********************************************************
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
