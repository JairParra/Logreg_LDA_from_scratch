
#### TESTS FROM LOGISTIC REGRESSION #### 


ex = {1,2,3} 
ex_sets = list(itertools.combinations(ex,1))
for subset in ex_sets: 
    for col in subset: 
        print(col)
        

X_copy = pd.DataFrame(X_redwine.copy()) 
X_copy_col = pd.DataFrame()
X_copy_col[1] = X_copy[1]



### 3. TESTING LOGISITC REGRESSION ### 
        
# subset for test


limit = int(len(X_redwine)*1)
X_train = X_redwine[:limit,:] # matrix
X_test = X_redwine[limit:,:]
y_train = y_redwine[:limit] 
y_test = y_redwine[limit:]  

cross_validation(LogisticRegression, X_train, y_train, 
                 folds=5, shuffle=False, random_state = 42, 
                     alpha_rate=0.002, auto_alpha=0.99, epochs=100)


new_X_red = X_redwine.copy() 

for col1 in T(X_redwine): 
    for col2 in T(X_redwine): 
        new_feat = np.multiply(col1, col2) 
        new_X_red = np.c_[new_X_red, new_feat]

cross_validation(LogisticRegression, new_X_red, y_redwine, epochs = 20,
                 folds=5, random_state= 1)
        

# Way # 1
logreg = LogisticRegression(X_train, y_train) # initialize 
logreg.cross_entropy_loss(verbose=False) # calculate loss 
logreg.gradient() # get gradient
final_loss = logreg.train(alpha=0.002, threshold=0.01, 
                          epochs=100, auto_alpha=1.0,
                          verbose=True) # run gradient descent and train the model 
logreg.plot_training_loss() # plot the training loss
logreg.predict_probabilities(X_train) # predict vector of probablities
y_pred = logreg.predict(X_test) # predict classifications
evaluate_acc(logreg, X_test, y_test, verbose=True)


# Way # 2
logreg = LogisticRegression() # instantiate 
logreg.fit(X_redwine,y_redwine,             # fit the model and train 
           alpha=0.002, threshold=0.001,
           epochs=100, auto_alpha=0.99, 
           verbose=False)
final_loss = logreg.cross_entropy_loss()    # obtain the final loss after training
logreg.predict_probabilities(X_test) # can check probabilities of the model 
y_pred = logreg.predict(X_test) # predict classifications
print("y_pred ", y_pred)
print("y_new ", list(y_test))
evaluate_acc(logreg, X_test, y_test, verbose=True)

# We can plot the training loss! 
logreg.plot_training_loss()





# ********************************************************************************

# second dataset 

limit = int(len(X_cancer)*0.9)
X_train = X_cancer[:limit,:] # matrix
X_test = X_cancer[limit:,:]
y_train = y_cancer[:limit] 
y_test = y_cancer[limit:]  

cross_validation(LogisticRegression, X_cancer, y_cancer, epochs = 5, folds=5)


# Way # 
logreg = LogisticRegression(X_train, y_train) # initialize 
logreg.cross_entropy_loss(verbose=False) # calculate loss 
logreg.gradient() # get gradient
final_loss = logreg.train(alpha=0.002, threshold=0.01, epochs=10, auto_alpha=0.99, verbose=True) # run gradient descent and train the model 
logreg.predict_probabilities(X_train) # predict vector of probablities
y_pred = logreg.predict(X_test) # predict classifications
evaluate_acc(logreg, X_test, y_test, verbose=True)


# ********************************************************************************
    
### 4. Testing LDA ### 
    
# PLEASE INPUT LDA TESTS IN HERE ONCE LDA IS COMPLETE  
    
# ********************************************************************************
    
### 5. Cross validation, parameter search, evaluation and performance ### 
    
        
        
    
