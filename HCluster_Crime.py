# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 12:26:29 2020

@author: RAVI
"""


import pandas as pd
import matplotlib.pylab as plt


crime2 = pd.read_csv("C:/RAVI/Data science/Assignments/DM-UnsupervisedLearning-Hierarchical Clustering/crime_data.csv/crime_data.csv")

def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return(x)
    
df_norm1= norm_func(crime2.iloc[:,1:])
df_norm1

from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch

z=linkage(df_norm1,method='complete',metric='euclidean')

from sklearn.cluster.hierarchical import AgglomerativeClustering
import sklearn.cluster.hierarchical as shch

plt.figure(figsize=(15,5));plt.title('Hierarchical Clustering Dendogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(z,leaf_rotation=0,leaf_font_size=8)

plt.show()

h_clustering=AgglomerativeClustering(n_clusters=6,affinity="euclidean",linkage="complete").fit(df_norm1)
h_clustering

h=pd.Series(h_clustering.labels_)
h

crime2['clust']=h
crime2=crime2.iloc[:,[5,0,1,2,3,4]]

crime2.iloc[:,2:].groupby(crime2.clust).median()

crime2.to_csv("crime2.csv",encoding="utf-8")

import os
os.getcwd()
