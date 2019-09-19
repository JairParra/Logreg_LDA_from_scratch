# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 12:16:12 2019

@author: jairp
"""

### *** COMP 551: Gradient Descent Implementation *** ### 

import numpy as np 
from numpy import transpose as T
from tqdm import tqdm

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


# 
np.random.seed(42)
X = np.random.rand(11,10) # (n x m)
y = np.random.rand(11,1) # (n x 1)
w = np.random.rand(10,1) # (m x 1)

for i in range(len(X)): 
    print(X[i])

print(np.matmul(X,w).shape)

X_new = np.random.rand(100,10)
X_0 = np.ones((100,1)) 
X_new = np.c_[X_0, X_new]

# Fit the data 
test = LinearRegression(X,y)
print(test.predict(X_new))
error = test.MSE() 
print(error)

print(type(test.gradient()))
print(test.gradient()*2)

test = LinearRegression(X,y)
test.train(alpha = 0.002, epochs=10)
