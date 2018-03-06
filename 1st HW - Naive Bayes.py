# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 22:32:21 2018

@author: Dusica
"""
import pandas as pd
import numpy as np
from scipy.stats import norm 
from pandas.api.types import is_numeric_dtype

def learn(data,label = None,alpha = 0.001):
    
    if(type(label) != str):
        label = data.columns[-1]
        
    apriori = data[label].value_counts()
    apriori += alpha
 
    apriori = apriori / (apriori.sum()) #+ alpha * len(data[label])
    apriori = np.log(apriori)
    model = {}
    model['apriori'] = apriori
    for atribute in data.drop(label,axis = 1).columns:
        model[atribute] = {}
        if(is_numeric_dtype(data[atribute])):
            for c in data[label].unique(): 
                subset = data[data[label]==c][atribute]
                mean_num=subset.mean()
                std_num=subset.std()
                model[atribute][c] = (mean_num,std_num)
#        II nacin
#            mean = data.groupby([label]).mean()[atribute]
#            std = data.groupby([label]).std()[atribute]
#            model[atribute] = [mean,std]
        else:
            freq_matrix = pd.crosstab(data[atribute],data[label]) 
            freq_matrix += alpha
            cont_matrix = freq_matrix / (freq_matrix.sum())# + alpha * len(data[atribute]))
            model[atribute] = np.log(cont_matrix) 
    return model

    
def predict(model, new):
   prediction={} 
   for row in range(new.shape[0]):
      prediction[row] = {}    
      for label_class in model['apriori'].index:
        probability = model['apriori'][label_class] 
        for attr in new.columns:
                if(is_numeric_dtype(data[attr])):
                    probability+=np.log(norm.pdf(new[attr][row],model[attr][label_class][0],model[attr][label_class][1]))
                    continue
                probability += model[attr][label_class][new[attr][row]]
        prediction[row][label_class] = np.exp(probability)
   return prediction

def predictRealProb(model, new):
    a = predict(model,new)
    print(a)
    all_pred_values=[]
    z=[]
    for row in a.keys():
       z.append(sum(a[row].values()))
       print('---------------------------')
       print(row,': ')
       print('Prediction of new: ')
       all_values=[]
       for k in a[row].keys(): #k is type of label
           a[row][k] = a[row][k] / z[row]
           print(k,a[row][k])
           all_values.append(a[row][k])
           all_labels=[]
           for label,value in a[row].items():
                all_labels.append(label)
       all_pred_values.append(all_labels[np.argmax(all_values)])
    
       print('Type of new predicted value is: ',all_labels[np.argmax(all_values)])
    print('\nFINAL predicted value/s is/are: ',all_pred_values)
    

#POZIV METODA
#%% DRUG
data = pd.read_csv("C:\\Users\\Dusica\\Desktop\\master aktuelno\\RAMU\\1. NaiveBayes\\1. NaiveBayes\\Domaci zadatak\\drug.csv")
model=learn(data,'Drug',alpha=0.00000001)
new = pd.read_csv("C:\\Users\\Dusica\\Desktop\\master aktuelno\\RAMU\\1. NaiveBayes\\1. NaiveBayes\\Domaci zadatak\\novi.csv")
predict(learn(data,'Drug',alpha=0.00000001),new)
predictRealProb(model,new)
#%% KREDIT

data = pd.read_csv("C:\\Users\\Dusica\\Desktop\\master aktuelno\\RAMU\\1. NaiveBayes\\1. NaiveBayes\\Kod\\kredit.csv")
new = pd.read_csv("C:\\Users\\Dusica\\Desktop\\master aktuelno\\RAMU\\1. NaiveBayes\\1. NaiveBayes\\Kod\\kredit_novi.csv")
predictRealProb(learn(data,alpha=0.00001),new)

#model=learn(data,alpha=0.00001)
#%% PREHLADA


data = pd.read_csv("C:\\Users\\Dusica\\Desktop\\master aktuelno\\RAMU\\1. NaiveBayes\\1. NaiveBayes\\Kod\\prehlada.csv")
new = pd.read_csv("C:\\Users\\Dusica\\Desktop\\master aktuelno\\RAMU\\1. NaiveBayes\\1. NaiveBayes\\Kod\\prehlada_novi.csv")
predict(learn(data,alpha=0.00001),new)
predictRealProb(learn(data,alpha=0.00001),new)

