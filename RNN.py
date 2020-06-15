# -*- coding: utf-8 -*-
"""
Created on Thu May  7 23:13:19 2020

@author: kingslayer
"""

#####  RECURRENT NEURAL NETWORKS   ##########


#PART 1(Data Preprocessing)

#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing training_set
dataset_train=pd.read_csv("Google_Stock_Price_Train.csv")
training_set=dataset_train.iloc[:,1:2].values

#feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler(feature_range=(0,1))
training_set_scaled=sc.fit_transform(training_set)

#creating datastructure with 20 timesteps
X_train=[]
y_train=[]
for i in range(60,1258):
    X_train.append(training_set_scaled[i-60:i,0])
    y_train.append(training_set_scaled[i])
X_train,y_train=np.array(X_train),np.array(y_train)

#Reshaping
X_train=np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))


#PART 2(Building The RNN)

#importing the libraries
from keras.layers import Dense,Dropout,LSTM
from keras.models import Sequential

#Initialising the RNN
regressor=Sequential()

#Adding first LSTM layer and dropout layer
regressor.add(LSTM(units=50,return_sequences=True,input_shape=(X_train.shape[1],1)))
regressor.add(Dropout(0.2))

#Adding two more layers
regressor.add(LSTM(units=50,return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=50,return_sequences=False))
regressor.add(Dropout(0.2))

#Adding the output layer
regressor.add(Dense(units=1))

#Compiling the RNN
regressor.compile(optimizer="rmsprop",loss="mean_squared_error")

#Fitting the RNN
regressor.fit(X_train,y_train,batch_size=32,epochs=100)


#PART 3(Predicting and Visualisation)

#importing real stock prices of 2017
dataset_test=pd.read_csv("Google_Stock_Price_Test.csv")
real_stock_price=dataset_test.iloc[:,1:2].values

#getting the predicted stock price
dataset_total=pd.concat((dataset_train["Open"],dataset_test["Open"]),axis=0)
inputs=dataset_total[len(dataset_total)-len(dataset_test)-60:].values
inputs=inputs.reshape(-1,1)
inputs=sc.transform(inputs)

X_test=[]
for i in range(60,80):
    X_test.append(inputs[i-60:i,0])
X_test=np.array(X_test)
X_test=np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))

predicted_stock_price=regressor.predict(X_test)
predicted_stock_price=sc.inverse_transform(predicted_stock_price)

#Visualisation
plt.plot(real_stock_price,color="red",label="real stock price")
plt.plot(predicted_stock_price,color="blue",label="predicted stock price")
plt.legend()
plt.show()

