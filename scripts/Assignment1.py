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

import math
import scipy
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns # easier & prettier visualization 
from tqdm import tqdm # to display progress bar
from numpy import transpose as T # because it is a pain in the ass
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

### 4. Helper functions ### 

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
        print("y_new :".format(list(y_new)))
        print("Accuracy: {}".format(acc))

    return acc         

# *****************************************************************************

### 3. TESTING LOGISITC REGRESSION ### 

# import the class from the script
from LogisticRegression import LogisticRegression


        
# subset for test
X_new = X_redwine[0:10,:] # matrix
y_new = y_redwine[0:10] # vector 

# new vector(test) 
x_new = X_new[0:1,:] # vector
x_new.shape

        
# Way # 1
        
logreg = LogisticRegression(X_redwine, y_redwine) # initialize 
logreg.cross_entropy_loss(verbose=False) # calculate loss 
logreg.gradient() # get gradient
final_loss = logreg.train(alpha=0.002, threshold=0.001, epochs=100, auto_alpha=0.99, verbose=False) # run gradient descent and train the model 
logreg.predict_probabilities(X_new) # predict vector of probablities
y_pred = logreg.predict(X_new) # predict classifications
print("y_pred ", y_pred)
print("y_new ", list(y_new))
evaluate_acc(logreg, X_new, y_new, verbose=True)




# Way # 2
logreg = LogisticRegression() # instantiate 
logreg.fit(X_redwine,y_redwine,             # fit the model and train 
           alpha=0.002, threshold=0.001,
           epochs=100, auto_alpha=0.99, 
           verbose=False)
final_loss = logreg.cross_entropy_loss()    # obtain the final loss after training
logreg.predict_probabilities(X_new) # can check probabilities of the model 
y_pred = logreg.predict(X_new) # predict classifications
print("y_pred ", y_pred)
print("y_new ", list(y_new))
evaluate_acc(logreg, X_new, y_new, verbose=True)

# We can plot the training loss! 
logreg.plot_training_loss()


# Parameter search algorithm: 
alphas = [0.002, 0.01, 0.05, 0.1]

losses = {}
accuracies = {}
# for each option for alpha 
for alpha in alphas: 
    # get the name of the alpha used
    model_name = "logreg_alpha_{}: \n".format(alpha)
    # instantiate the model
    model = LogisticRegression() 
    # fit and train the model with the appropriate X and y
    model.fit(X_redwine, y_redwine) 
    # obtain the loss of the model 
    loss = model.cross_entropy_loss() 
    # store in the losses dictionary  
    losses[model_name] = loss 
    # obtain the predictions on (new) test validation set
    y_pred = model.predict(X_new)
    # calculate accuracy 
    acc = (y_pred == y_new).sum() / len(y_pred)*100 
    # store in accuracies dictionary 
    accuracies[model_name] = acc
    print("\n")
    

# ********************************************************************************
    
### 4. Testing LDA ### 
    
# PLEASE INPUT LDA TESTS IN HERE ONCE LDA IS COMPLETE  
    
# ********************************************************************************
    
### 5. Cross validation, parameter search, evaluation and performance ### 
    

X = np.random.rand(10,3) 
X = pd.DataFrame(X)  
    
def cross_split(X,folds=5): 
    
    # for each split"
    for k in range(folds): 
        
        # display split number
        print("\nSplit {}\n".format(k+1))
        
        split = X.shape[0] / folds # obtain splitting index
        upper = int(split*(k+1)) # obtain upper index
        lower = max(int(upper - split),0) # obtain lower index 
        
        print("lower = {}".format(lower))
        print("upper = {}".format(upper)) 
        
        X_lower = X.iloc[:lower,:]  # obtain lower train set
        X_val = X.iloc[lower:upper,:]  # obtain validation set 
        X_upper = X.iloc[upper:,:]  # obtain upper train set
        
        # stack lower train and upper train together
        X_train = X_lower.append(X_upper) 
        
        print("X_train: ", X_train) 
        print("\nX_val :{} \n".format(X_val))
        


def kfold_CV_search(algorithm, X, y, k = 5, params={}, random_state=42): 
    """ 
    Performs a k-fold Cross-validation search testing the given parameters, 
    which have to be input as a dictionary, where the keys are the 
    values of the parameters to test and the values are lists which include 
    a range of parameters to test. 
    
    params = {"alpha" :[0.001, 0.002, 0.003], 
              "threshold" : [0.001, 0.01, 0.1], 
              "auto_alpha" : [0.99, 0.95, 0.90]
               } 

    params: 
        @ algorithm: an instance of a classification algorithm (log reg) 
        @ X: input training features data
        @ y: input targets data  
        @ k: number of folds to perform 
        @ params: dictionary of parameters to test
        @ random_state: seed for the random 
    """
    
    random_state = random_state 
    X = pd.DataFrame(X) 
    X = X.iloc[np.random.permutation(len(X))] # randomly shuffle the data 

    return X 
    
    
    
    
