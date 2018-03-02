# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 22:32:21 2018

@author: Dusica
"""
import pandas as pd
import numpy as np
from scipy.stats import norm 
from pandas.api.types import is_numeric_dtype
#data = pd.read_csv("C:\\Users\\Dusica\\Desktop\\master aktuelno\\RAMU\\1. NaiveBayes\\1. NaiveBayes\\Kod\\kredit.csv")
#data['Ozenjen'].value_counts()
#freq_matrix = pd.crosstab(data['Ozenjen'],data['Vratio']) 
#freq_matrix / freq_matrix.sum() #+ len(data['Ozenjen'].unique())
data = pd.read_csv("C:\\Users\\Dusica\\Desktop\\master aktuelno\\RAMU\\1. NaiveBayes\\1. NaiveBayes\\Domaci zadatak\\drug.csv")
#alpha = 1 je da nam bude definisn log...bar malo epsilon da vrati vrednos ili min float smooothing
def learn(data,label = None,alpha = 0.001):
    #is_numeric=pd.DataFrame(t in ['int64','float64'] for t in data2.dtypes)
    #np.where(is_numeric==True)
    
    if(type(label) != str):
        label = data.columns[-1]
    alpha = 0.001
    apriori = data[label].value_counts()
    apriori += alpha
    #dodaj malo broj float min!
    apriori = apriori / (apriori.sum())# + alpha * len(data[label]))
    apriori = np.log(apriori)
    model = {}
    model['apriori'] = apriori
    for atribute in data.drop(label,axis = 1).columns:
        model[atribute] = {}
        if(is_numeric_dtype(data[atribute])):
            for c in data[label].unique(): 
                subset = data[data[label]==c][atribute]
                mean_num=subset.std()
                std_num=subset.mean()
                model[atribute][c] = (mean_num,std_num)
        else:
            freq_matrix = pd.crosstab(data[atribute],data[label]) 
            freq_matrix += alpha
            cont_matrix = freq_matrix / (freq_matrix.sum())# + alpha * len(data[atribute]))
            model[atribute] = np.log(cont_matrix) 
    return model

new = pd.read_csv("C:\\Users\\Dusica\\Desktop\\master aktuelno\\RAMU\\1. NaiveBayes\\1. NaiveBayes\\Domaci zadatak\\novi.csv")
    #new = new.iloc[0]
    #new['Ozenjen'][0]
print(learn(data))
def predict(model, new):
    prediction = {}
    for label_class in model['apriori'].index:
        probability = model['apriori'][label_class]
        for attr in new.columns:
            if(is_numeric_dtype(data[attr])):
                probability+=np.log( norm.pdf(new[attr],data[attr][0],data[attr][1]))
                continue
            probability += model[attr][label_class][new[attr][0]]
        prediction[label_class] = np.exp(probability)
    return prediction

print(predict(learn(data,alpha=0.00001),new))
a = predict(learn(data,alpha=0.00001),new)
print(a)
z =  sum(a.values())
#a/sum
print('Prediction of new: ')
for k in a.keys():
    a[k] = a[k] / z
    print(k,a[k])
    #print(k[np.argmax(a.values())])
#if(pd.dtype(data['Ozenjen']))
#data.dtypes