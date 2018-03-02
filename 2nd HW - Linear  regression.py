import pandas as pd
import numpy as np
np.random.seed(1)
np.set_printoptions(formatter={'float_kind':lambda x: "%.3f" % x})
    
#%% PREPARE DATA
#data = pd.read_csv('C:\\Users\\Dusica\\Downloads\\LR podaci\\house.csv')

#%%
#ubaci label
def prepareData(data):
    y = data.iloc[:,[-1]]
    X = data.iloc[:,0:-1]
    
    X_mean = X.mean()
    X_std = X.std()
    X = (X - X_mean) / X_std
    
    X['X0'] = 1
     
    y = y.as_matrix()
    X = X.as_matrix()
    n,m = X.shape
    
# INITIALIZATION
    w = np.random.random((1,m))
    alpha = 0.06
    lambda_penalty=10
    return X,y,n,m,w,alpha,lambda_penalty
    #%% LEARN: 
    # stochastic (incremental) gradient descent 
    #OVO 
def stochastic(data):   
    X,y,n,m,w,alpha,lambda_penalty=prepareData(data)
    for iter in range(10):
        print(iter,'Epoch')
        np.random.shuffle(X)    
        for i in range((X.shape[0])):
           pred = (pd.DataFrame(X).iloc[i,:]).as_matrix().dot(w.T)
           err = pred-y[i]
           grad = err*(pd.DataFrame(X).iloc[i,:].as_matrix())#+(lambda_penalty*w)
           w = w - alpha * grad
           #grad_norm = abs(grad).sum()
           
           #if grad_norm<0.1: 
#           print(str(i)+' row', grad_norm)
#           pred = X[0:i].dot(w.T)
#           err = pred-y[0:i]
#           grad = err.T.dot(X[0:i]) / (i+1)   
#           mean_square_error = err.T.dot(err) / (i+1)    
#           print('First '+str(i+1)+' rows:',mean_square_error)   
        pred = X.dot(w.T)      
        err = pred-y    
        grad = err.T.dot(X) / n      
        mean_square_error = err.T.dot(err) / n   
        print('Whole dataset:',mean_square_error)
        print(w)
        
           #break
    #return w   
        
           #if grad_norm<0.1 or mean_square_error<10: break
        #mean_square_error = err*(err)
       
        

    #%% LEARN: GRADIENT DESCEND with lambda penalty batch
def learn(data):
    X,y,n,m,w,alpha,lambda_penalty=prepareData(data)
    for iter in range(10000):
       pred = X.dot(w.T)
       err = pred-y
       reg=2*(lambda_penalty*w)
       reg[-1]=0
       grad = (err.T.dot(X) / n )+reg
       w = w - alpha * grad
       mean_square_error = err.T.dot(err) / n
       grad_norm = abs(grad).sum()
       if grad_norm<0.1 or mean_square_error<10: break
       print(iter, grad_norm, mean_square_error)
    return w
    
    #%% PREDICT
    #data_new = pd.read_csv('C:\\Users\\Dusica\\Downloads\\LR podaci\\house_new.csv')
    #data_new = (data_new-X_mean)/X_std
    #data_new['X0'] = 1
    #prediction = data_new.as_matrix().dot(w.T)
    #print(prediction)
data = pd.read_csv('C:\\Users\\Dusica\\Desktop\\master aktuelno\\RAMU\\2. Linear regression\\2. Linear regression\\Domaci\\Boston_Housing.txt',sep='\t')
learn(data)
#stochastic(data)
#data = pd.read_csv('C:\\Users\\Dusica\\Downloads\\LR podaci\\house.csv')
#stochastic(data)