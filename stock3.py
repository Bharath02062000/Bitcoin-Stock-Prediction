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

print(training_data_len)

#Scale the data to makeit more standardised into a range of values
scaler = MinMaxScaler(feature_range=(0,1)) #declare the mixmax scalar functionality....
final_dataset=new_data.values

print(final_dataset)

