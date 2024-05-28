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

#fetch the column names of the dataframe
print(data_Frame.columns)

#fetch the column type of the data frame
print(data_Frame.dtypes)

#fetch detailed informations of the data frame
print(data_Frame.info(verbose=True))

#fetch the raw data in the data frames
print(data_Frame.values)

#fetch the memory usage of the columns of the data frame
print(data_Frame.memory_usage())


#Visualize the closing price hisory of the company by fetching the live Bitcoin
plt.figure(figsize=(16,8))
plt.title('Close Price History for the company')
plt.plot(data_Frame['Close'])
plt.xlabel('Data', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.show()
