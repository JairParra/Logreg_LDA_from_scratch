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
        col_avg = np.mean(df[column]) 
        col_std = np.std(df[column]) 
        df[column] = df[column].apply(lambda x: ((x-col_avg) / col_std) ) 
        

# Get column names 
red_wine_df_cols = list(red_wine_df.columns)[:-1] # don't include the targets

# Pairplot without normalization 
sns.pairplot(red_wine_df.drop('quality', axis= 1))

# Normalize
normalize_df(red_wine_df, columns=red_wine_df_cols)

# Pairplot after normalization 
sns.pairplot(red_wine_df.drop('quality', axis= 1))

# Shape and statistics
print("Red wine df shape: {}".format(red_wine_df.shape))  
red_wine_df_stats = red_wine_df.drop('quality', axis=1).describe()

# Plot the distributions
red_wine_good = red_wine_df['quality'][red_wine_df['quality'] == 'Good'].count()
red_wine_bad = red_wine_df['quality'][red_wine_df['quality'] == 'Bad'].count()
sns.countplot(red_wine_df['quality'])

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
cancer_df = cancer_df.dropna().drop('id', axis=1) # drop na rows
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
cancer_df_stats = cancer_df.drop('Class', axis=1).describe()

# Plot pairplot
df = cancer_df.drop('Class',axis=1)
sns.pairplot(df)

# Plot the label distributions: 
sns.countplot(cancer_df['Class'])

# Since the all the features seem to lie in similar ranges, we will not
# normalize them. 

# *****************************************************************************
























