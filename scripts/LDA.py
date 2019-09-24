# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 21:44:41 2019

@ COMP 551 : Applied Machine Learning (Winter 2019)
@ Mini-project 1 : Implementing Logistic Regression and LDA from scratch

# Team Members: 
@ Hair Albeiro Parra Barrera
@ ID: 260738619 

@ Sun Gengyi 
@ ID: 260768270
    
@ Shu hao
@ ID: 

@author: sungengyi
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
from numpy.linalg import inv

### 2. Implementing LDA ###
class LDA:
    np.random.seed(42)
    np.set_printoptions(precision = 2)

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
      self.m = X.shape[1] # Because of intercept term 
      self.X = X  
      self.w = np.random.rand(self.m, 1) # initialize weights 
      self.y = y 
       
      # initialize number of data points from class 1 and class 0 
      self.N1 = 0
      self.N0 = 0
      
      #initialize P(y)
      self.PY0 = 0
      self.PY1 = 0
      
      #initialize mean vectors     
      self.miu_1 = np.zeros(self.m)
      self.miu_0 = np.zeros(self.m)
      
      #initialize covariance
      self.covariance = np.zeros((self.m,self.m))
        
   
    def find_probability(self):
        '''
        Count the probability of each binary features
        '''
        # initialize to 0 
        index = 0
        # number of data points equal to 0
        N0 = 0    
        # number of data points equal to 0
        N1 = 0
        # increment variables
        for element in self.y:
            index+=1
            if element == 0:
                N0+=1
            elif element == 1:
                N1+=1
            else:
                # return error function if element does not fall in nerither condition
                print("ERROR: Binary Data ONLY" )
        
        
        # Assign values to attributes
        self.N0 = N0
        self.N1 = N1
        self.PY0 = N0/self.n
        self.PY1 = N1/self.n
        

              
    def find_miu(self):
        # miu_0 = sigma from i to n ( select(yi = 0) * xi) / N0
        # miu_1 = sigma from i to n ( select(yi = 1) * xi) / N1

        '''
        Calculate the mean vectors for y_i = 0 and y_i = 1
        '''
        # initailize 
        miu_0 = np.zeros(self.m)
        miu_1 = np.zeros(self.m)
        
        # initialize the loop counter -> index
        index = 0
        #loop through yi 
        for element in self.y:
            # sum x_i according to class (class 0 or class 1)
            if element == 0:
                miu_0 = np.add(miu_0,self.X[index])
            elif element == 1:
                miu_1 = np.add(miu_1,self.X[index])
            else:
                #Error message inidicates not well-cleaned data
                print("ERROR: Binary Data ONLY")
            # increment counter
            index+=1
        # divide the sumation by number of data points in its class
        self.miu_0 = np.multiply(miu_0, 1/self.N0)
        self.miu_1 = np.multiply(miu_1, 1/self.N1)

  
    def find_covariance(self):
        
        # loop counter
        index = 0
        # a temporary matrix used to store (x_i - miu_k) i -> 1:n, k->1:0 
        temp_matrix = np.zeros((self.m,self.m))
        # an all-zero matrix
        covariance = np.zeros((self.m,self.m))
        for element in self.y:
            
            # calculate (x_i - miu_k) i -> 1:n, k->1:0
            # calculate it times its transpose matrix
            # add the result to the covariance matrix
            if element == 0 :
                temp_matrix[0] = np.subtract(self.X[index],self.miu_0)
                covariance = np.add(covariance,np.matmul(T(temp_matrix),temp_matrix))
            elif element == 1 :
                temp_matrix[0] = np.subtract(self.X[index],self.miu_1)
                covariance = np.add(covariance,np.matmul(T(temp_matrix),temp_matrix))

            else:
                print("ERROR: Binary Data ONLY")
            # increment index
            index+=1
        # unbiased covariance matrix
        self.covariance = np.multiply(covariance,1/(self.N0+self.N1-2))

    def find_log_odds(self,X_input):
        x_n = X_input[0].shape
        index = 0
        log_odds = 0
        y_output = np.zeros(x_n) 

        first_term = np.log(test.PY1/test.PY0)  
        second_term = 1/2 * np. dot(np.dot(T(self.miu_1),inv(self.covariance)),self.miu_1)
        third_term = 1/2 * np. dot(np.dot(T(self.miu_0),inv(self.covariance)),self.miu_0)
        forth_term_part_2 = np.subtract(self.miu_1,self.miu_0)

        for element in X_input:
            forth_term_part_1 = np.dot(T(X_input[index]),inv(self.covariance))
            forth_term = np.dot(forth_term_part_1,forth_term_part_2)
            # calculate the log odds for each entity
            log_odds = np.add(np.add(first_term,-second_term),np.add(third_term,forth_term))
            # classify according to log odds ratio
            # if the result > 0, -> class 0, -> class 1 otherwise
            if log_odds > 0:
                y_output[index] = 1
            elif log_odds <= 0:
                y_output[index] = 0
            else: 
                print("ERROR: Log odds ratio cannot be processed")
            index+=1
        # output the binary resuult
        return y_output
            

    def fit(self):
        # calculate N1, N0, PY1, PY0, miu_1, miu_0
        # and its covariance matrix
        # store these values in the instance
        self.find_probability()
        self.find_miu()
        self.find_covariance()
        
    def predict(self, X_input):
         '''
         call find log-odds, assign 0/1 to data sets according to decision boundary
         '''
         return self.find_log_odds(X_input)


        
            

#These are testing codes

X_test = np.array([[5,2,3,4],
                  [0,3,1,7],
                  [7,0,1,2],
                  [2,7,1,9]])
X_input = np.array([[4,2,3,4],
                  [0,3,1,7],
                  [7,0,1,2],
                  [2,7,1,9]])
Y_test = np.array([[1],
                  [1],
                  [1],
                  [0]])

test = LDA(X_test,Y_test)
test.find_probability()
test.find_miu()
test.find_covariance()

print(test.find_log_odds(X_input))

print("test_miu0",test.miu_0)
print("test_miu1",test.miu_1)
print("test covariance",test.covariance)
print(test.predict(X_input))






    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    