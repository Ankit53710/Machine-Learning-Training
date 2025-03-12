import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as seabornInstance
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LinearRegression

dataset =pd.read_csv('weather.csv')
print(dataset.shape)
print(dataset.describe())

dataset.plot(x='MinTemp',y='MaxTemp',style='o')
plt.title('MinTemp Vs MaxTemp')
plt.xlabel('MinTemp')
plt.ylabel('MaxTemp')
plt.show()

plt.figure(figsize=(15,10))
plt.tight_layout()
seabornInstance.displot(dataset['MaxTemp'])
plt.show()

#data splicing
x=dataset['MinTemp'].values.reshape(-1,1)
y=dataset['MaxTemp'].values.reshape(-1,1)

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=0)

model=LinearRegression()
model.fit(xtrain,ytrain)#training of data

#intercept and coefficient

print('Intercept',model.intercept_)
print('coefficient',model.coef_)

y_pred=model.predict(xtest)

df=pd.DataFrame({'Actaul':np.array(ytest).flatten(),'Predicted':np.array(y_pred).flatten()})

print(df)

df1=df.head(10)

df1.plot(kind='bar',figsize=(16,10))

plt.grid(which='major',linestyle='-',linewidth='0.5',color='green')

plt.grid(which='minor',linestyle=':',linewidth='0.5',color='Black')

plt.show()

plt.scatter(xtest,ytest,color='gray')

plt.plot(xtest,y_pred,color='red',linewidth=2)
plt.show()

#Errors printing

#mean_absolute_error
print('mean_absolute_error',metrics.mean_absolute_error(ytest,y_pred))

#mean_squared_error
print('mean_squared_error',metrics.mean_squared_error(ytest,y_pred))

#Root_mean_squared_error

print('root_mean_absolute_error',np.sqrt(metrics.mean_absolute_error(ytest,y_pred))) 