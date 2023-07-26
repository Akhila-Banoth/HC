#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Hierarchical Clustering PROJECT


# In[3]:


#Import the libraries
import pandas as pd
import matplotlib.pyplot as plt


# In[6]:


#Import dataset
dataset = pd.read_csv("Mall_Customers.csv")
X = dataset.iloc[:, :].values


# In[7]:


X


# In[9]:


#Dendrogram to find optimal number of clusters
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title("Dendrogram")
plt.xlabel("Customers")
plt.ylabel("Euclidean distance")
plt.show()


# In[16]:


#Train the model
from sklearn.cluster import AgglomerativeClustering
clustering = AgglomerativeClustering(n_clusters = 3)
y_hc = clustering.fit_predict(-X)


# In[17]:


y_hc


# In[18]:


#Visualising the clusters
plt.scatter(X[y_hc == 0 , 0] ,X[y_hc == 0 , 1], c = 'red' , label = 'Cluster 1' )
plt.scatter(X[y_hc == 1 , 0] ,X[y_hc == 1 , 1], c = 'green' , label = 'Cluster 2' )
plt.scatter(X[y_hc == 2 , 0] ,X[y_hc == 2 , 1], c = 'pink' , label = 'Cluster 3' )
#plt.scatter(X[y_hc == 3 , 0] ,X[y_hc == 3 , 1], c = 'blue' , label = 'Cluster 4' )
#plt.scatter(X[y_hc == 4 , 0] ,X[y_hc == 4 , 1], c = 'orange' , label = 'Cluster 5' )
plt.title("Cluster of Customers")
plt.xlabel("Annual Income(K$)")
plt.ylabel("Spending Score(1-100)")
plt.legend()
plt.show()


# In[ ]:




