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

import time
import math
import scipy
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns # easier & prettier visualization 
from tqdm import tqdm # to display progress bar
from numpy import transpose as T # because it is a pain in the ass
from LogisticRegression import LogisticRegression # the class we implemented
sns.set()

# *****************************************************************************

### 2. Loading the Data, Preprocessing  & Statistics ### 

## 2.1 Load the red wine dataset
red_wine_df = pd.read_csv('../data_raw/winequality-red.csv', sep = ';') 

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
print("Red wine 'Good' counts: ", red_wine_df['quality'][red_wine_df['quality'] == 1].count() )
print("Red wine 'Bad' counds: ", red_wine_df['quality'][red_wine_df['quality'] == 0].count() )
red_wine_df_stats = red_wine_df.drop('quality', axis=1).describe()

# Label distribution plot
sns.countplot(red_wine_df['quality'])
plt.savefig('../figs/redwine_countplot.png')

# Features pairplot
sns.pairplot(red_wine_df.drop('quality', axis= 1), diag_kind='kde')
plt.savefig('../figs/redwine_pairplot.png')



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
cancer_df =  pd.read_csv('../data_raw/breast-cancer-wisconsin.data') # read data
cancer_df.columns = cancer_cols  # rename columns 
cancer_df = cancer_df.dropna().drop('id', axis=1) # drop na rows and id
cancer_df = cancer_df[cancer_df['Bare Nuclei'] != '?'] # drop rows with missing vals 
cancer_df['Bare Nuclei'] = pd.to_numeric(cancer_df['Bare Nuclei']) # change back to int
cancer_df.loc[cancer_df['Class'] == 4, 'Class' ] = 0 # malignant
cancer_df.loc[cancer_df['Class'] == 2, 'Class' ] = 1 # benign

# Fix data type for each column in feature set
for column in cancer_df.columns[:-1]: 
    print(column)
    cancer_df[column] = pd.to_numeric(cancer_df[column])
    
# normalize columbns
normalize_df(cancer_df, columns = cancer_df.columns[:-1])

# Print shape and obtain statistics. 
print("cancer_df shape: {}".format(cancer_df.shape)) 
print("cancer data 'Benign' counts: ", cancer_df['Class'][cancer_df['Class'] == 1].count() )
print("cancer data 'Malignant' counts: ", cancer_df['Class'][cancer_df['Class'] == 0].count() )
cancer_df_stats = cancer_df.drop('Class', axis=1).describe()

# Labels distribution plot 
fig = sns.countplot(cancer_df['Class'])
plt.title('Cancer dataset Class countplot')
plt.savefig('../figs/cancer_countplot.png')

# Features pairplot
sns.pairplot(cancer_df.drop('Class',axis=1), diag_kind='kde')
plt.savefig('../figs/cancer_pariplot.png')

# Since the all the features seem to lie in similar ranges, we will not
# normalize them. 


## 2.3 Save and load data 


# Save
red_wine_df.to_csv('../data_clean/redwine_df.csv', index=False) 
cancer_df.to_csv('../data_clean/cancer_df.csv', index=False)

# Load 
red_wine_df = pd.read_csv('../data_clean/redwine_df.csv') 
cancer_df = pd.read_csv('../data_clean/cancer_df.csv')


## 2.4 Separate both dataset into features and labels


# Red wine data features and labels
X_redwine = red_wine_df.drop('quality', axis=1).values.astype('float64')
y_redwine = red_wine_df['quality']

# Cancer data features and labels 
X_cancer = cancer_df.drop('Class', axis=1).values.astype('float64')
y_cancer = cancer_df['Class']



# *****************************************************************************

### 4. Util functions ### 

def evaluate_acc(model, X_test, y_test, verbose=True): 
    """
    Evaluates model accuracy and returns the score for 
    a binary classification model. The input model should 
    have a function called predict()
    params: 
        @ model: the input classification model 
        @ X_test: Input 
    """
    y_pred = model.predict(X_test) 
    acc = (y_pred == y_test).sum() / len(y_pred) * 100
    
    if verbose: 
        print("y_pred :{}".format(y_pred)) 
        print("y_new :".format(list(y_pred)))
        print("Accuracy: {}".format(acc))

    return acc      


def cross_validation(input_model, X,y,folds=5, shuffle=True, random_state = 42, 
                     alpha_rate=0.002, auto_alpha=0.99, epochs=100, verbose=False):
    
    np.random.seed(random_state)
    X = pd.DataFrame(X)  # convert to dataframe
    accuracies = np.zeros(folds)

    
    if shuffle: 
        X = X.iloc[np.random.permutation(len(X))] # randomly shuffle the data        
    
    # for each split
    for k in range(folds): 
                
        # display split number
        print("\nSplit {}\n".format(k+1))
        
        split = X.shape[0] / folds # obtain splitting index
        upper = int(split*(k+1)) # obtain upper index
        lower = max(int(upper - split),0) # obtain lower index 
        
        X_lower = X.iloc[:lower,:]  # obtain lower feature train set
        X_val = X.iloc[lower:upper,:]  # obtain val feature set
        X_upper = X.iloc[upper:,:]  # obtain upper feature train set
        
        y_lower = y[:lower] # obtain lower targets 
        y_val = y[lower:upper]  # obtain validation targets
        y_upper = y[upper:] # obtain upper targets 
        
        # stack lower train and upper train together
        X_train = np.array(X_lower.append(X_upper))
        y_train = np.r_[y_lower, y_upper]
        
        # instantiate the model 
        model = input_model()
        
        print("Type X_train: ", type(X_train))
        print("shape X_train: ", X_train.shape)
        print("Type X_train: ", type(y_train))
        print("shape X_train: ", y_train.shape)
        
        
        # Test for instance of Logistic Regression 
        if isinstance(model, LogisticRegression):         
            # fit (and train) the model 
            model.fit(X_train, y_train, alpha=alpha_rate, epochs = epochs, 
                      threshold= 0.01, auto_alpha=auto_alpha, verbose=verbose) 
            
#            model.plot_training_loss()
            
            # obtain accuracy  
            acc = evaluate_acc(model, X_val, y_val)
            # append to the accs list 
            accuracies[k] = acc
            
            
        print("accuracies: ", accuracies) 
        print("\n mean accuracy: ", np.mean(accuracies))
        
    return np.mean(accuracies) 
        

# *****************************************************************************
    
### 3. Running Logistic Regression Model Experiments ### 
    
## 3.1 : Testing different learning rates for Logistic Regression 
    
alphas = [1, 0.2, 0.02, 0.002, 0.0002]

def test_alphas(X, y, alphas=[]): 
    """
    Runs cross-validation using different input alphas. 
    params: 
        @ X: matrix of features
        @ y: target vector
        @ alphas: list of different learning rates to try
    returns: 
        @ accuracies: a dictionary with the accuracies for each alpha
    """
    accuracies = {}
    
    for alpha in alphas: 
        name = "cv_acc_alpha_{}".format(alpha) 
        accuracies[name] = cross_validation(LogisticRegression, X, y, 
                 folds=5, shuffle=False, random_state = 42, 
                     alpha_rate=alpha, auto_alpha=0.99, epochs=100)
        
    return accuracies 
        

accs = test_alphas(X_redwine, y_redwine, alphas=alphas) 
print(accs)

# We find that alpha = 0.002 yields the best accuracy. 

## 3.1 Comparing Log Reg and LDA running times

# Logistic Regression 
t0 = time.time() 
logreg = LogisticRegression() # instantiate 
logreg.fit(X_redwine,y_redwine,             # fit the model and train 
           alpha=0.002, threshold=0.001,
           epochs=100, auto_alpha=0.99, 
           verbose=False)
t1 = time.time() 
logreg_time = t1 - t0 


# LDA 
t0 = time.time() 
# LDA model running goes here 
t1 = time.time() 
LDA_time =  t1 - t1 
    
# Compare    
print("Logistic regression running time: {} s".format(logreg_time))
print("LDA running time: {} s".format(logreg_time))



    


