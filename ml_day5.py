# -*- coding: utf-8 -*-
"""ML day5

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1oeNtnhuNOS3n4DAaPESYdxUhA47zFdAR

**explicit intelligence**

bagging & boosting for over fitting(ensemble learning)
"""

from sklearn.datasets import load_breast_cancer
dataset=load_breast_cancer()
x=dataset.data
y=dataset.target
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2)

from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
from sklearn.ensemble import BaggingClassifier
BG=BaggingClassifier(base_estimator=model,n_estimators=100,max_features=10,max_samples=100)
BG.fit(xtrain,ytrain)#training

ypred_test=BG.predict(xtest)
ypred_train=BG.predict(xtrain)

from sklearn.metrics import accuracy_score
print(accuracy_score(ytest,ypred_test))
print(accuracy_score(ytrain,ypred_train))

"""**without Bagging**"""

model.fit(xtrain,ytrain)
y_test=model.predict(xtest)
y_train=model.predict(xtrain)
print(accuracy_score(ytest,y_test))
print(accuracy_score(ytrain,y_train))

"""**K Fold Cross Validation**"""

from sklearn.model_selection import cross_val_score
score=cross_val_score(model,x,y,cv=5)#value of k=cv
print(score.mean()) #mean for average

"""**Unsupervised learning**

types=Association rule minning,clustering(recommendation system,location of earthquick,finding epicenter),self organizining map

euclidean distance
"""

from sklearn.datasets import make_blobs
x,y=make_blobs(n_samples=1500,n_features=2,centers=3,cluster_std=1.5)#x,y are features & total cluster

import matplotlib.pyplot as plt
plt.scatter(x[:,0],x[:,1])

from sklearn.cluster import KMeans
wcss=[]
k=[1,2,3,4,5,6,7,8,9]
for i in k:
  km=KMeans(n_clusters=i)
  km.fit(x)
  wcss.append(km.inertia_)

plt.plot(k,wcss)

km=KMeans(n_clusters=3)
ypred=km.fit_predict(x)
ypred

plt.scatter(x[:,0],x[:,1],c=ypred)

#silhouette score between -1 to 1