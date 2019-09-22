# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 13:50:48 2019

@author: jairp
"""

### *** cross split implementation *** ###

import numpy as np 
import pandas as pd

X = np.random.rand(10,3) 
X = pd.DataFrame(X)   

def cross_split(X,folds=5): 

    # obtain number of folds to split on 
    folds = folds
    
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
        X_test = X.iloc[lower:upper,:]  # obtain validation set 
        X_upper = X.iloc[upper:,:]  # obtain upper train set
        
        # stack lower train and upper train together
        X_train = X_lower.append(X_upper) 
        
        print("X_train: ", X_train) 
        print("\nX_val :{} \n".format(X_val))
        
    
cross_split(X)
    