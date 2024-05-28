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
#Show the data
print(data_Frame)

# making boolean series for a team name 
new_df = data_Frame[(data_Frame.High>14)]
print(new_df)

# fetch the top data from the data frame
print(data_Frame.head)

# Fetch the row and  column count of data frames
print(data_Frame.shape)

# fetch the bottom layer of data frame
print(data_Frame.tail)
