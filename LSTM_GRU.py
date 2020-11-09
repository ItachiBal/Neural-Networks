# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 11:38:06 2020

@author: Itachi Bal(PHANINDRA BALAJI)
"""
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import os


print(os.listdir("C:/Users/Itachi Bal/Desktop/ML/project"))#returns the content in this following address
bit_data=pd.read_csv("C:/Users/Itachi Bal/Desktop/ML/project/bitstampUSD_1-min_data_2012-01-01_to_2020-09-14.csv")#read data
bit_data.head()#display first five rows
bit_data["date"]=pd.to_datetime(bit_data["Timestamp"],unit="s").dt.date #converting the time stamp to date-time
group=bit_data.groupby("date") # grouping the data according to new column date 
data=group["Close"].mean()
data.shape #(3178,)shape of the data
data.isnull().sum() #0 checks for null values in the data
data.head()#display first five rows
print(bit_data.columns)#name of the columns

# data[1:367]
# data[368:(368+364)]
# data[368+364:(368+364+365)]
# plt.plot(data[368+364:(368+364+365)])
# plt.plot(data)
# plt.xlabel("YEAR")
# plt.ylabel("Close-price")
# plt.show()
##########################################
#eda

#GOAL: predict the closing price of the bitcoin of a following day
close_train=data.iloc[:len(data)-50] # training set
close_test=data.iloc[len(close_train):] # testing set
 

plt.figure(figsize=(15,5))
plt.plot(np.log(close_train))# as the data follows a bit of exponential distribution, log tranformation of the data gives us better scope
plt.xlabel("PERIOD")
plt.ylabel("Close-price")
plt.title("LOG-TRANSFORMATION OF CLOSE PRICE")
plt.show()
# plt.figure(figsize=(15,5))
# plt.plot(close_test)
# plt.xlabel("PERIOD")
# plt.ylabel("Close-price")
# plt.show()

# plt.figure(figsize=(10, 6))
# plt.plot(np.log(close_test))
# plt.xlabel("LOG-TRANSFORMATION OF CLOSE_PRICE")
# plt.show()

max(close_train)
min(close_train)
np.mean(close_train)
#feature scaling (set values between 0-1)
close_train=np.array(close_train)
close_train=close_train.reshape(close_train.shape[0],1) #converting the 1-dim array to column vector
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
close_scaled=scaler.fit_transform(close_train)

#training scheme
#create a set of linear rows(0-50) as X-TRAIN
#and 51 as Y-TRAIN
timestep=50
x_train=[] # creating x & y data
y_train=[]

for i in range(timestep,close_scaled.shape[0]): #training scheme
    x_train.append(close_scaled[i-timestep:i,0])
    y_train.append(close_scaled[i,0])

x_train,y_train=np.array(x_train),np.array(y_train)
x_train=x_train.reshape(x_train.shape[0],x_train.shape[1],1) #reshaping the input for RNN
# print("x_train shape= ",x_train.shape)
# print("y_train shape= ",y_train.shape)


#################SIMPLE-RNN##################
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, Dropout,Flatten#importing necessary files

regressor=Sequential()#starting the sequential model
#first RNN layer
regressor.add(SimpleRNN(128,activation="relu",return_sequences=True,input_shape=(x_train.shape[1],1)))
regressor.add(Dropout(0.25))#dropout layer
#second RNN layer
regressor.add(SimpleRNN(256,activation="relu",return_sequences=True))#here 256 is the output of the hidden layer
regressor.add(Dropout(0.25)) # 25 percent of the inputs are dropped
#third RNN layer
regressor.add(SimpleRNN(512,activation="relu",return_sequences=True))
regressor.add(Dropout(0.35))

#fourth RNN layer
regressor.add(SimpleRNN(256,activation="relu",return_sequences=True))
regressor.add(Dropout(0.25))
#fifth RNN layer
regressor.add(SimpleRNN(128,activation="relu",return_sequences=True))
regressor.add(Dropout(0.25))
#convert the matrix to 1-line
regressor.add(Flatten())# flatten the output
#output layer
regressor.add(Dense(1))

# defining loss and optimizers
regressor.compile(optimizer="adam",loss="mean_squared_error")
regressor.fit(x_train,y_train,epochs=50,batch_size=55)#training the model


print(regressor.summary())#architecture of the rnn

#preparing X-TEST for prediction

inputs=data[len(data)-len(close_test)-timestep:] # last 100 values
inputs=inputs.values.reshape(-1,1)
inputs=scaler.transform(inputs)

x_test=[]
for i in range(timestep,inputs.shape[0]):
    x_test.append(inputs[i-timestep:i,0])
x_test=np.array(x_test)
x_test=x_test.reshape(x_test.shape[0],x_test.shape[1],1)

predicted_data=regressor.predict(x_test)
predicted_data=scaler.inverse_transform(predicted_data)

data_test=np.array(close_test)
data_test=data_test.reshape(len(data_test),1)

#comparing the prediction VS true with plots
plt.figure(figsize=(8,4), dpi=80, facecolor='w', edgecolor='k')
plt.plot(data_test,color="r",label="true result")
plt.plot(predicted_data,color="b",label="predicted result")
plt.legend()
plt.xlabel("Time(50 days)")
plt.ylabel("Close Values")

plt.grid(True)
plt.title("SIMPLE-RNN") 
plt.show()
from sklearn.metrics import mean_squared_error

MSE = mean_squared_error(data_test, predicted_data)
MSE
# different parameters
#epochs batch-size MSE
#100 55 517171.28
#100 61 918243.01
#100 64 1996415.42
#100 70 2479293.138
#100 30 1452989.71
###############LSTM-RNN#################
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout,Flatten

model=Sequential()
model.add(LSTM(27,input_shape=(None,1),activation="relu")) # LSTM layer
model.add(Dense(1))
model.compile(loss="mean_squared_error",optimizer="adam")
model.fit(x_train,y_train,epochs=500,batch_size=35) #training the model
print(model.summary())
#preparing the X-TEST for prediction
inputs=data[len(data)-len(close_test)-timestep:]
inputs=inputs.values.reshape(-1,1)
inputs=scaler.transform(inputs)

x_test=[]
for i in range(timestep,inputs.shape[0]):
    x_test.append(inputs[i-timestep:i,0])
x_test=np.array(x_test)
x_test=x_test.reshape(x_test.shape[0],x_test.shape[1],1)

predicted_data=model.predict(x_test)
predicted_data=scaler.inverse_transform(predicted_data)

data_test=np.array(close_test)
data_test=data_test.reshape(len(data_test),1)

#plots showing prediction VS true

plt.figure(figsize=(8,4), dpi=80, facecolor='w', edgecolor='k')
plt.plot(data_test,color="r",label="true result")
plt.plot(predicted_data,color="b",label="predicted result")
plt.legend()
plt.xlabel("Time(50 days)")
plt.ylabel("Close Values")
plt.grid(True)
plt.title("LSTM-RNN")
plt.show()

MSE_LSTM = mean_squared_error(data_test, predicted_data)
MSE_LSTM #MSE
#################################################
#different parameters tested
#LSTM-20 cells-50 epochs-25 batch size- 54640.69 MSE
#LSTM-30 cells-50 epochs-25 batch size- 63175.51 MSE
#LSTM-40 cells-50 epochs-25 batch size- 95204.07 MSE
#LSTM-20 cells-100 epochs-25 batch size-50882.67  MSE
#LSTM-20 cells-100 epochs-35 batch size-51206.13  MSE
#LSTM-27 cells-200 epochs-30 batch size-70301.64  MSE
#LSTM-27 cells-500 epochs-35 batch size-49441.44  MSE

#####################GRU-RNN###################
from keras.models import Sequential
from keras.layers import Dense, Dropout,Flatten,GRU

model1=Sequential()

model1.add(GRU(27,input_shape=(None,1),activation="relu"))#GRU layer

model1.add(Dense(1))

model1.compile(loss="mean_squared_error",optimizer="adam")

model1.fit(x_train,y_train,epochs=500,batch_size=35)#training the model

model1.summary()

#preparing X-TEST
inputs=data[len(data)-len(close_test)-timestep:]
inputs=inputs.values.reshape(-1,1)
inputs=scaler.transform(inputs)

x_test=[]
for i in range(timestep,inputs.shape[0]):
    x_test.append(inputs[i-timestep:i,0])
x_test=np.array(x_test)
x_test=x_test.reshape(x_test.shape[0],x_test.shape[1],1)

#predicting
predicted_data=model1.predict(x_test)
predicted_data=scaler.inverse_transform(predicted_data)

data_test=np.array(close_test)
data_test=data_test.reshape(len(data_test),1)
#plotting predicted VS true values
plt.figure(figsize=(8,4), dpi=80, facecolor='w', edgecolor='k')
plt.plot(data_test,color="r",label="true result")
plt.plot(predicted_data,color="b",label="predicted result")
plt.legend()
plt.xlabel("Time(50 days)")
plt.ylabel("Close Values")
plt.grid(True)
plt.title("GRU-RNN")
plt.show()

MSE_GRU = mean_squared_error(data_test, predicted_data)
np.sqrt(MSE_GRU)#RMSE
#different parameters tested
#GRU-20 cells-50 epochs-25 batch size- 49992.78 MSE
#GRU-27 cells-200 epochs-30 batch size- 43293.28, 42553.65 MSE
#GRU-27 cells-500 epochs-35 batch size- 43611.81 MSE
#GRU-25 cells-100 epochs-35 batch size- 46960.89 MSE
#GRU-30 cells-50 epochs-25 batch size- 50434.04 MSE
#GRU-40 cells-50 epochs-25 batch size- 70712.49 MSE
#GRU-20 cells-100 epochs-25 batch size- 54855.80 MSE
#GRU-20 cells-100 epochs-35 batch size- 68216.09 MSE
#######################################################
# x_test=[]
# for i in range(timestep,inputs.shape[0]):
#     x_test.append(inputs[i-timestep:i,0])
# x_test=np.array(x_test)
# x_test=x_test.reshape(x_test.shape[0],x_test.shape[1],1)
# inputs
# predicted_data
# predicted_data[0:50,0]
# predicted_data=model1.predict(x_test)
# predicted_data=scaler.inverse_transform(predicted_data)
# data_test=np.array(close_test)
# data_test=data_test.reshape(len(data_test),1)
# ip=predicted_data
# ip=scaler.transform(ip)
# x_predicted_test=[]
# for i in range(1):
#     x_predicted_test.append(ip[0:50,0])
# x_predicted_test=np.array(x_predicted_test)
# x_predicted_test=x_predicted_test.reshape(x_predicted_test.shape[0],x_predicted_test.shape[1],1)
# new_predicted_data=model1.predict(x_predicted_test)
# new_predicted_data=scaler.inverse_transform(new_predicted_data)
# adding_data=new_predicted_data
# x_predicted_test=[]
# for i in range(1):
#     x_predicted_test.append(ip[1:50,0])
#     x_predicted_test.extend([0.5465725])
# x_predicted_test=np.array(x_predicted_test)
# x_predicted_test=x_predicted_test.reshape(x_predicted_test.shape[0],x_predicted_test.shape[1],1)











