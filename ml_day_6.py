# -*- coding: utf-8 -*-
"""Ml day 6

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1dnc8kVfFt2R6EkyRYDdqRaV3xMejn2jk

**Transform learning**

imagenet(dataset)-VGG
"""

base='/content/sample_data/flower1'
import os
folders=os.listdir(base)
folders

from google.colab import drive
drive.mount('/content/drive')

import cv2                   #  open CV is a image libeary in which we can perform many operations using open cv
x=[]
y=[]

for i in folders:
  for j in os.listdir(base+'/'+i):
    a=j.split('.')
    if a[1] == 'jpg':
      p = cv2.imread(base+'/'+i+'/'+j)
      r=cv2.resize(p,(224,224))             #  VGG net isi size ki image ko input me leta h
      x.append(r)
      y.append(i)           #   this is a folder name which we use as LABEL

import numpy as np
xarr=np.array(x)
yarr=np.array(y)
xarr=xarr/255.0
#xarr.shape



for i in range(len(y)):
  if y[i]=='Rose':
    y[i]=0
  else:
    y[i]=1
y

from keras.applications.vgg16 import VGG16

vgg=VGG16(input_shape=(224,224,3),weights='imagenet',include_top=False)#top layer removed
vgg.summary()

vgg.trainable=False
vgg.summary()

from keras.models import Sequential
from keras.layers import Flatten,Dense
model=Sequential()
model.add(vgg)
model.add(Flatten())
model.add(Dense(1,activation='sigmoid'))
model.summary()

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
from keras.callbacks import EarlyStopping,ModelCheckpoint
es=EarlyStopping(min_delta=0.001,monitor='accuracy',patience=3,mode='auto')
mcp=ModelCheckpoint('model.hdf5',save_best_only=False)

o=model.fit(xarr,yarr,epochs=10,validation_data=(xarr,yarr),batch_size=2,callbacks=[es,mcp])

from keras.models import load_model
mymodel=load_model('/content/model.hdf5')

import matplotlib.pyplot as plt
plt.imshow(xarr[0])

output=mymodel.predict(xarr)
output

from math import e
o1=output>0.5

base1='/content/sample_data/flower2'
import os
folder=os.listdir(base)
folder

import cv2                   #  open CV is a image libeary in which we can perform many operations using open cv
xtest=[]
ytest=[]

for i in folders:
  for j in os.listdir(base+'/'+i):
    a=j.split('.')
    if a[1] == 'jpg':
      p = cv2.imread(base+'/'+i+'/'+j)
      r=cv2.resize(p,(224,224))             #  VGG net isi size ki image ko input me leta h
      xtest.append(r)
      ytest.append(i)           #   this is a folder name which we use as LABEL

import numpy as np
xtestarr=np.array(x)#shape is for array converting in array
ytestarr=np.array(y)
xtestarr=xarr/255.0
xtestarr.shape

o=model.fit(xtestarr,ytestarr,epochs=10)

o=model.predict(xtestarr)
o=np.array(o)
print(o)
print(ytestarr)
#from sklearn.metrics import accuracy_score
#accuracy_score(ytestarr,o)

"""**Early stopping** in keras"""

#day 4
#day 3 use earlystopping

#ways of taking errors
#strocasting(draw=its to slow,jeekjecking(more filacchuation)),
#batch(draw=taking to much time after taking batch error)
#,minibatch(best one,taking 25 sample in batch)