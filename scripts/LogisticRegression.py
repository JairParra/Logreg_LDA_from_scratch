# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 13:35:01 2019

@author: jairp
"""

### *** Logistic Regression algorithm *** ### 

import math
import scipy
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns # easier & prettier visualization 
from tqdm import tqdm # to display progress bar
from numpy import transpose as T # because it is a pain in the ass



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
        self.w = np.random.rand(self.m, 1) # randomly initialize weights 
        self.y = y 
        self.last_train_losses = [0] # this will be used to plot training loss
        
        print("Initialized with dimensions\n X:({}) y:({})".format(self.X.shape, self.y.shape)) 
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
        print(input_shape)
        
        # If input is a vector 
        if X_new.shape[0] == 1: 
            # If input length doesn't match  
            if (X_new.shape[1] + 1) != self.m: 
                message = "Input number of features doesn't match model number of parameters" 
                message += "\nInput is has {} features but model has {} features".format(len(X_new), self.m - 1)
                raise Exception(message)
            else: 
                print("HERE")
                x = np.insert(X_new, 0, 1) # insert an extra one at the beginning
                wTx = float(np.matmul(T(self.w), x)) 
                sigm_wTx = self.sigmoid(wTx) 
                print("sigm_wTx", sigm_wTx)
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
            
            if verbose:
                print("wTx: " , wTx) 
                print("sigm_wTx ", sigm_wTx)
                print("log(sigm_wTx) ", math.log(sigm_wTx))
            
            if y_i == 1:
                total_loss.append(-1*math.log(sigm_wTx + 0.0001)) 
            else: 
                total_loss.append(1*math.log(1-sigm_wTx + 0.0001))
            
            
        loss = -1*np.sum(np.array(total_loss))
        
        if verbose: 
            print("Model loss: ", loss)

        return loss
    
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
    
    def train(self, alpha=0.002, threshold=0.001, epochs=100, auto_alpha=0.99, verbose=True): 
        """
        Trains the model using the fitted feature matrix X and target vector y using 
        gradient descent. 
        params: 
            @ alpha: learning rate 
            @ threshold: threshold used in early stopping 
            @ epochs: number of (maximum) iterations for training 
            @ auto_alpha: auto_adjusts the learning rate in every iteration 
            @ verbose: display training progress
        returns: 
            @ final_loss 
            @ losses: list of all loss values during training
        """
        
        # initialize error
        prev_loss = self.cross_entropy_loss()
        initial_loss = prev_loss
        losses = [] # will make part of the object
        
        if verbose: 
            # display training progress
            for k in tqdm(range(epochs), desc="\nTraining...") : 
                
                grad = alpha*self.gradient() # calculate gradient   
                temp = np.add(self.w, grad) # get weights update
                self.w = temp # update weights
                loss = self.cross_entropy_loss() # calculate current error
                
                print("\nEpoch {}".format(k+1))
                print("Cross-entropy loss: %.2f" % (loss) )
                            
                # early stopping
                if abs(loss-prev_loss) < threshold: 
                    break
                
                losses.append(loss)
                
                prev_loss = loss
                alpha = auto_alpha*alpha
                
        else: 
            for k in range(epochs): 
                
                grad = alpha*self.gradient() # calculate gradient   
                temp = np.add(self.w, grad) # get weights update
                self.w = temp # update weights
                loss = self.cross_entropy_loss() # calculate current error
                            
                # early stopping
                if abs(loss-prev_loss) < threshold: 
                    break
                
                losses.append(loss)
                
                prev_loss = loss
                alpha = auto_alpha*alpha
            
            
        print("---Terminated---")  
        
        final_loss = prev_loss
        print("Initial loss: {}".format(initial_loss)) 
        print("Final loss: {}".format(final_loss))
        
        self.last_train_losses = losses
        
        return final_loss        
    
    
    def plot_training_loss(self): 
        """
        Plots all the losses generated during training loss
        """ 
        x = range(len(self.last_train_losses)) 
        y = self.last_train_losses 
        plt.figure() 
        plt.plot(x,y, label = "Training loss") 
        plt.xlabel("Epochs") 
        plt.ylabel("Loss") 
        plt.title("Training loss through time")
        plt.legend() 
        plt.show() 
    
    def fit(self, X,y,alpha=0.02, threshold=0.001, epochs=100, auto_alpha=0.99, verbose=False): 
        """
        Initializes the model with the input parameters 
        and trains
        """
        self.__init__(X,y) # Initialize with input 
        self.train(alpha, threshold, epochs, auto_alpha, verbose)
