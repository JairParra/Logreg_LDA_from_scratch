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
    
@ Hao Shu
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
import itertools 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns # easier & prettier visualization 
from tqdm import tqdm # to display progress bar
from numpy import transpose as T # because it is a pain in the ass
from LogisticRegression import LogisticRegression # the class we implemented
from LDA import LDA
from scipy.stats import stats
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
sns.countplot(cancer_df['Class'])
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

### 3. Util functions ### 

def shuffle_data(X,y, random_state=42): 
    """
    Randomly shuffles the input targets and features by joining them into 
    a dataframe, shuffling it and then recovering back the pairwise 
    shuffled inputs. 
    """
    
    np.random.seed(random_state) # initialize seed
    X = pd.DataFrame(X)  # convert to dataframe
    y = pd.DataFrame(y)
    df = pd.concat([X,y], axis=1) # concatenate into 1 
    df = df.iloc[np.random.permutation(len(df))] # shuffle randomly
    X = df.iloc[:,:-1]  # recover feature set
    y = np.array(df.iloc[:,-1:]) # recover target vector
    y = y.reshape((y.shape[0],)) # convert back to series format
    
    return X,y


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
    
    if verbose: 
        print("(input) X.shape = ", X.shape) # input X
        print("(input) y.shape = ", y.shape) # input y
    
    if shuffle: 
        X, y = shuffle_data(X,y)

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
        
        if verbose: 
            print("shape X_train: ", X_train.shape)
            print("shape y_train: ", y_train.shape)
            print("shape X_val: ", X_val.shape)
            print("shape y_val: ", y_val.shape)
        
        
        # Test for instance of Logistic Regression 
        if isinstance(model, LogisticRegression):         
            # fit (and train) the model 
            model.fit(X_train, y_train, alpha=alpha_rate, epochs = epochs, 
                      threshold= 0.01, auto_alpha=auto_alpha, verbose=verbose) 
            
#            model.plot_training_loss()
            
            # obtain accuracy  
            acc = evaluate_acc(model, X_val, y_val, verbose)
            
            # append to the accs list 
            accuracies[k] = acc
            
        else:  
            
            # fit and train the model 
            X_train = np.array(X_train) 
            y_train = np.array(y_train) 
            X_val = np.array(X_val) 
            y_val = np.array(y_val)
            
            model.fit(X_train, y_train)
            
            # obtain accuracy 
            acc = evaluate_acc(model, X_val, y_val, verbose) 
            
            # append to the accs list
            accuracies[k] = acc 
            
            
        mean_acc = np.mean(accuracies)
            
        print("accuracies: ", accuracies) 
        print("\n mean accuracy: ", mean_acc)
        
    return np.mean(accuracies) 
        

# *****************************************************************************
    
### 4. Running Tests and Experiments ### 
    
## 4.1 : Testing different learning rates for Logistic Regression 
    
alphas = [1, 0.2, 0.02, 0.002, 0.0002]

def CV_search(X, y, alphas=[], epochs_list=[100], folds = 5): 
    """
    Runs cross-validation using different input alphas and epochs. 
    params: 
        @ X: matrix of features
        @ y: target vector
        @ alphas: list of different learning rates to try
    returns: 
        @ accuracies: a dictionary with the accuracies for each alpha
    """
    accuracies = {}
    
    for alpha in alphas: 
        for epochs in epochs_list:
            name = "cv_acc_alpha=_{}_epoch={}".format(alpha, epochs) 
            accuracies[name] = cross_validation(LogisticRegression, X, y, 
                     folds=folds, shuffle=True, random_state = 42, 
                         alpha_rate=alpha, auto_alpha=0.99, epochs=epochs)
        
    return accuracies 
        
# Redwine dataset 
redwine_accs = CV_search(X_redwine, y_redwine, alphas=alphas) 
print("Redwine accuracies: ", redwine_accs)
"""　Best: alpha 0.002 -> 74.35% """

# Cancer dataset 
cancer_accs = CV_search(X_cancer, y_cancer, alphas=alphas) 
print("Cancer accuracies: ", cancer_accs)
"""　Best: alpha 0.02 -> 96.92% """

## 4.2 Comparing best LogReg accuracy vs. best LDA accuracy 

# 4.2.1 Redwine Dataset 

# Logistic Regression 
cross_validation(LogisticRegression, X_redwine, y_redwine, shuffle=True, 
                 folds=5, alpha_rate=0.002, auto_alpha=0.99, epochs=100) 
""" 74.35% """

# LDA 
cross_validation(LDA, X_redwine, y_redwine, shuffle=True)
"""74.42"""


# 4.2.2 Cancer Dataset 

# Logistic Regression 
cross_validation(LogisticRegression, X_cancer, y_cancer, shuffle=True, 
                 folds=5, alpha_rate=0.2, auto_alpha=0.99, epochs=100) 
"""96.92%"""

# LDA 
cross_validation(LDA, X_cancer, y_cancer, shuffle=True, verbose=True)
"""95.90%"""


## 4.3 Comparing Log Reg and LDA running times


# Logistic Regression 
t0 = time.time() 
logreg = LogisticRegression() # instantiate 
logreg.fit(X_redwine,y_redwine, # fit and train the model 
           alpha=0.002,  # learning rate
           threshold=0.001, # early stopping threshold
           epochs=100,  # max number of epochs 
           auto_alpha=0.99,  # alpha rate 
           )
t1 = time.time() 
logreg_time = t1 - t0 


# LDA 
t0 = time.time() 
lda = LDA() 
lda.fit(X_redwine, y_redwine)
t1 = time.time() 
LDA_time =  t1 - t0
    
# Compare    
print("Logistic regression running time: {} s".format(logreg_time))
print("LDA running time: {} s".format(LDA_time))


## 3.2 Improving the accuracy of the wine dataset 



# Copy the dataset
def new_feats(X, only_quadratic=False, 
                 only_interactions=False,
                 all_interactions=False, 
                 logarithmic=False, 
                 correlation=0.6): 
    """
    Adds second order terms to the dataset: Either only quadratic terms, 
    only interaction terms (X_i != X_j) or both, if they correlation is 
    higher than a specified value. If 'logarithmic' is specified, 
    it converts the parameters into logarithmic values. 
    """
    
    # copy the original feature set
    new_X = X.copy()
    
    if logarithmic: 
        new_X = pd.DataFrame(new_X)
        for col in new_X: 
            new_X[col] = new_X[col].apply(lambda x: np.exp(x))
        new_X = np.array(new_X)
    
    for col1 in T(X): 
        for col2 in T(X): 
            
            if only_interactions: 
                if stats.pearsonr(col1, col2)[0] >= 0.6 and not np.array_equal(col1, col2): 
                    new_feat = np.multiply(col1, col2) 
                    new_X = np.c_[new_X, new_feat]  
                    
            elif only_quadratic: 
                if stats.pearsonr(col1, col2)[0] >= 0.6 and np.array_equal(col1, col2): 
                    new_feat = np.multiply(col1, col2) 
                    new_X = np.c_[new_X, new_feat]  
                    
            elif all_interactions: 
                if stats.pearsonr(col1, col2)[0] >= 0.6: 
                    new_feat = np.multiply(col1, col2) 
                    new_X = np.c_[new_X, new_feat]  
                
                    
    return new_X

# Obtain new feature sets
X_redwine_quadratic = new_feats(X_redwine, only_quadratic=True) 
X_redwine_only_interactions = new_feats(X_redwine, only_interactions=True) 
X_redwine_all = new_feats(X_redwine, all_interactions=True) 
X_redwine_e = new_feats(X_redwine, logarithmic=True)



# Run CV search for each of them  
quadratic_accs =  CV_search(X_redwine_quadratic, y_redwine, 
                            alphas=alphas, epochs_list=[50,100,150]) 

"""Best: cv_acc_alpha=_0.002_epoch=150 -> 74.17"""

only_interactions_accs = CV_search(X_redwine_only_interactions, y_redwine, 
                                   alphas=alphas, epochs_list=[50,100,150]) 

"""Best: cv_acc_alpha=_0.002_epoch=100 -> 74.35"""

all_accs = CV_search(X_redwine_all, y_redwine, 
                     alphas=alphas, epochs_list=[50,100,150])

"""Best: cv_acc_alpha=_0.002_epoch=150 -> 73.67"""

e_accs =  CV_search(X_redwine_e, y_redwine, 
                     alphas=alphas, epochs_list=[50,100,150])

"""Best: cv_acc_alpha=_0.0002_epoch=150 -> 63.66"""


# We see that even with different interaction parameters as well as 
# cross-validation grid search, we were unable to obtain a better accuracy. 

# Now we will try to test every possible subset with the same parameter
# configurations. 

def best_subset(X,y, verbose=False): 
    
    new_X = X.copy() # create a copy 
    new_X = pd.DataFrame(X) # convert to dataframe 
    new_X_cols = set(new_X.columns) # obtain set of columns
    accuracies = {} # dictionary to store 
    
    
    # for each subset size
    for k in range(len(new_X_cols)): 
        # obtain all subsets of size k 
        col_names_subsets = list(itertools.combinations(new_X_cols,k+2) )
        print(col_names_subsets)
        
        # for each subset of size k 
        for subset in col_names_subsets: 
            # obtain the name
            model_name = "subset_k={}_cols={}".format(k, str(subset))
            X_subset = pd.DataFrame() # initialize empty data frame
            print("\n{}\n".format(model_name))
            
            # obtain the columm number 
            for col_num in subset: 
                # append the column to a new data 
                X_subset[col_num] = new_X[col_num]
                                
            if verbose: 
                print(X_subset)
            # fit_train a model and get CV accuracy
            acc = cross_validation(LogisticRegression, 
                                   X_subset, y_redwine, 
                                   shuffle=True, 
                                   folds=5, 
                                   alpha_rate=0.002, 
                                   auto_alpha=0.99, 
                                   epochs=150,
                                   verbose=verbose) 
            
            # store that accuracy 
            accuracies[model_name] = acc
            
            
    return accuracies 


# accuracies for all possible subset of original dataset
best_origin_accs = best_subset(X_redwine, y_redwine)

for acc in best_origin_accs: 
    print(acc)
