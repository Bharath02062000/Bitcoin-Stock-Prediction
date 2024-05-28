#import libraries
import math
import pandas_datareader as DataReaderWeb
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM,Dropout
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

# Get the live Bitcoin cost quote
data_Frame = DataReaderWeb.DataReader('AAPL', data_source='yahoo', start='2011-01-01', end='2019-12-17')

#Create a new dataframe with only the 'Close','Date' columns
new_data = data_Frame.filter(['Close','Date'])
#Conver the dataframe to a numpy array
dataset = new_data.values
#get the number of rows to train the model on
training_data_len = math.ceil(len(dataset) * .8)


#Scale the data to makeit more standardised into a range of values
scaler = MinMaxScaler(feature_range=(0,1)) #declare the mixmax scalar functionality....
final_dataset=new_data.values


#Create the training dataset
#Create the scaled training dataset    0 1 2 3 4 5 6 7 8 9  =>10
train_data = final_dataset[0:training_data_len , :]   #   0 1 2 3 4 5 6 7  > 0.8 data
valid_data = final_dataset[training_data_len:,:]  # testing data .....#   8 9   =>0.2 data
#Split the data into x_train and y_train datasets
scaler=MinMaxScaler(feature_range=(0,1))
scaled_data=scaler.fit_transform(final_dataset)
x_train,y_train = [],[]
for i in range(60, len(train_data)):
  x_train.append(scaled_data[i-60:i,0])
  y_train.append(scaled_data[i, 0])
  if i<= 60:
    print(x_train)
    print(y_train)
    print()

    #Convert the x_train and y_train to numpy array
x_train, y_train = np.array(x_train), np.array(y_train)   # numpy array is used to do mathematical operations on  it....

#Reshape the data
x_train = np.reshape(x_train,(x_train.shape[0], x_train.shape[1], 1))   # data and its structure
x_train.shape

#Build the LSTM model
lstm_model=Sequential()
lstm_model.add(LSTM(units=50,return_sequences=True,input_shape=(x_train.shape[1],1)))
lstm_model.add(LSTM(units=50))
lstm_model.add(Dense(1))

inputs_data=new_data[len(new_data)-len(valid_data)-60:].values
inputs_data=inputs_data.reshape(-1,1)
inputs_data=scaler.transform(inputs_data)

X_test=[]
for i in range(60,inputs_data.shape[0]):
    X_test.append(inputs_data[i-60:i,0])
X_test=np.array(X_test)
X_test=np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
predicted_closing_price=lstm_model.predict(X_test)
predicted_closing_price=scaler.inverse_transform(predicted_closing_price)

lstm_model.save("saved_model.h5")

train_data=new_data[:training_data_len]
valid_data=new_data[training_data_len:]
valid_data['Predictions']=predicted_closing_price
#visualize the data
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date',fontsize=18)
plt.ylabel('Close Price USD ($)' , fontsize=18)
plt.plot(train_data["Close"])
plt.plot(valid_data[['Close',"Predictions"]])
plt.legend(['Close','Predictions'],loc='lower right')

plt.show()











