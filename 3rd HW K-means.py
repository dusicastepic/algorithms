# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 22:24:31 2018

@author: Dusica
"""
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist

def kmeans(data,ponder,dist_type,k):

    n,m=data.shape
    ponder=np.ones((1,m))
    ponder=ponder
    #K-means --0 korak set k
    data_mean=data.mean()
    data_std=data.std()
    data=(data-data_mean)/data_std
    inits=3
    
    #ponderisanje
    
    
    best_cluster_centroids=None
    best_cluster_quality=float('inf')
    best_assign=None
    #1. C=random, moze radnom seed da ne menja vre.svaki put
    
    for init in range(inits):
        centroids=data.sample(1)#.reset_index(drop=True) 
        data_copy=data.copy()
        for  cl in range(k-1):
        #while(True):        
            #sum_cen=[sum(cdist(data_copy,centroids)[cen]) for cen in range(len(cdist(data_copy,centroids)))]
            sum_cen=cdist(data_copy,centroids).sum(axis=1)
    
            centroids=centroids.append(data_copy.iloc[(np.argmax(sum_cen)),:])
            #data.drop(data_copy.index[(np.argmax(sum_cen))])

            data_copy.drop(data_copy.index[(np.argmax(sum_cen))],inplace=True)
            centroids=centroids.reset_index(drop=True)
            
            #centroids.drop_duplicates(inplace=True)
            #if centroids.shape[0]==k:
             #   break
        
    #    centroids=centroids.append(data.iloc[(np.argmax(cdist(data,centroids))),:])
    #    #centroids.append(data.reset_index(drop=True).iloc[(np.argmax(cdist(data,centroids))),:])
    #    np.argmax(cdist(data,centroids)[:,0]+cdist(data,centroids)[:,1])
    #    
    #    centroids=centroids.reset_index(drop=True)
        
        assign=np.zeros((n,1)) #matrica
        quality=np.zeros((k))
        old_quality=float('inf')
        data=data*ponder[0]
        #korak 2. min distance Xi od Ck
        #tacka cddist udaljenija od svih centroiida
        for iter in range(20):
            for i in range(n):
#                point=data.iloc[i,:]
                #point=point*ponder[0]
                if dist_type==0:
                    dist=cdist(data,centroids,'euclidean')
                    #dist=((point-centroids)**2).sum(axis=1)#odavde nadalje za zahtev 2
                    #assign[i]=np.argmin(dist)
                if dist_type==1:
                        dist=cdist(data,centroids,'cityblock')
                    #dist=abs((point-centroids)).sum(axis=1)#odavde nadalje za zahtev 2
                if dist_type==2:
                        dist=cdist(data,centroids,'mahalanobis', VI=None)
                assign[i]=np.argmin(dist[i])
                
                
#                dist=((point-centroids)**2).sum(axis=1)#odavde nadalje za zahtev 2
#                assign[i]=np.argmin(dist)
                
        #ovde gore smo samo dodelili tacke najbliziima
        #VARIJANSA KLASTERA, minimiziramo varijansu klstera  hehe sr.jvadratno odstupanje        
            for c in range(k):
            #    subset=data[assign==c]
             #   centroids.iloc[c,:]=subset.mean()
                centroids.iloc[c,:]=data[assign==c].mean()
                quality[c]=data[assign==c].var().sum()*len(data[assign==c])
          
    
                if quality.sum()==old_quality:break
        #    if abs(quality.sum()-old_quality)<=0.1:break
                 #print(iter,quality.sum(),quality)    
                 #print(centroids)
                
                old_quality=quality.sum()  
                #print(old_quality)
            

        
            if old_quality<=best_cluster_quality:
                  
                best_cluster_quality=old_quality
                best_cluster_centroids=centroids
                best_assign=assign
                #data['best_assign']=assign
#        print(old_quality)
#    print('Best model')    
#    print('Best quality--->',best_cluster_quality)    
#    print('--Best cluster centroids--\n',best_cluster_centroids)    
    return best_cluster_centroids,best_assign,data
    #calculateSI(k,best_cluster_centroids,best_assign,data)
    
def determineK():
   numberOfK=range(3,10)
   all_si=[]
   for k in numberOfK:
       centroids,assign,dat=kmeans(data,[1,1,2,1,1,1,1,1,1,2,1,1,1,1],0,k)
       si=calculateSI(k,centroids,assign,dat)
       all_si.append(si)
       print('Number of clusters: ',k,'Silhouette index',si)

   print('Best Number of clusters: ',numberOfK[np.argmax(all_si)],'Silhouette index',all_si[np.argmax(all_si)])

   #kmeans(data,[1,1,2,1,1,1,1,1,1,2,1,1,1,1],0)
   #print(k,'Sillhuete score for ',str(k),' clusters is')   
   #kmeans(data,[1,1,2,1,1,1,1,1,1,2,1,1,1,1],2)# min siluete
#determineK()   

def calculateSI(k,centroids,assign,data):
    # mean intra-cluster distance
#    centroids,assign,data=kmeans(data,[1,1,2,1,1,1,1,1,1,2,1,1,1,1],0)
    a=[]
    b=[]
#    data_copy=data.copy()
#    data_copy['best_assign']=assign.astype(int)
    #data_copy['best_assign'] = data_copy['best_assign'].astype(int)
    #data=data.reset_index(drop=True)
    for cluster in range(k):
        a.append(cdist(data[assign==cluster],data[assign==cluster]).mean())
        
#        II NACIN
#         dp=0
#        for datapoint in range(data[assign==cluster].shape[0]):
#            #a.append((data_copy[data_copy.best_assign==cluster]-centroids.iloc[cluster,:]).sum(axis=0).mean())
#            #dp=(data.iloc[datapoint,:]-(data[assign==cluster])).sum(axis=1).mean()
#            dp+=cdist(pd.DataFrame(data[assign==cluster].iloc[datapoint,:]).transpose(),data[assign==cluster]).mean()
#        a.append(dp/data[assign==cluster].shape[0])
#            
    #mean nearest-cluster distance
#    II nacin
#        sum_cen=cdist(data[assign==cluster],centroids.drop(centroids.index[cluster])).sum(axis=0)
        sum_cen=cdist(pd.DataFrame(centroids.iloc[cluster,:]).transpose(),centroids.drop(cluster))

#
        nearest_cluster=np.argmin(sum_cen)
        
    #for cluster in range(k):
#        dp=0
            #a.append((data_copy[data_copy.best_assign==cluster]-centroids.iloc[cluster,:]).sum(axis=0).mean())
            #dp=(data.iloc[datapoint,:]-(data[assign==cluster])).sum(axis=1).mean()
        
        b.append(cdist(data[assign==cluster],data[assign==nearest_cluster]).mean())
        si=[]
        for l in range(len(a)):
            maks=[]
            maks.append(a[l])
            maks.append(b[l])
            maks=np.max(maks)
            si.append((b[l]-a[l])/maks)
        
    si=np.array(si).sum()/len(si)
#    print('Number of clusters: ',k,'Silhouette index',si)
    return si
        #dp+=cdist(data[assign==cluster],data[assign==nearest_cluster]).sum(axis=1).mean()
        #b.append(dp/data[assign==cluster].shape[0])
        
#            b.append((data[assign==cluster]-centroids.iloc[nearest_cluster,:]).sum(axis=0).mean())
#            c=[]
#            si=[]
#            c.append(a[cluster])
#            c.append(b[cluster])
#            si+=(b[cluster]-a[cluster])/np.max(c)

            #si=(np.mean(b)-np.mean(a))/np.max(a,b)
#            print(si)
#    print(si/k)

    
#POZIV METODE    
data = pd.read_csv('C:\\Users\\Dusica\\Desktop\\master aktuelno\\RAMU\\2. Linear regression\\2. Linear regression\\Domaci\\Boston_Housing.txt',sep='\t')
# 0 Euclidean, 1 Manhattan/City block
#
#kmeans(data,[1,1,2,1,1,1,1,1,1,2,1,1,1,1],0)
#dist_type=0

#data=pd.read_csv('C:\\Users\\Dusica\\Downloads\\life.csv').set_index('country')
