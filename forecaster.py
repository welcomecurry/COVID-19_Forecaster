import pandas as pd
import numpy as np
from datetime import datetime
import math
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from keras.preprocessing.sequence import TimeseriesGenerator

# read data
corona_data = pd.read_csv('/home/amit/Desktop/covidForecaster/time_series_covid19_confirmed_global_04_03_20.csv')

# remove unnecessary data
corona_data.drop(labels = ['Province/State', 'Lat', 'Long'], axis=1, inplace=True)

# shape data to represent dates as indexes and column names to country names
corona_data.shape

# group data of countries since there are repeats
corona_data.groupby(['Country/Region']).sum()
corona_data.drop(labels = ['Country/Region'], axis=1, inplace=True)

# reshape data for time series analysis
corona_data_reshape = pd.DataFrame()

for i in range(0, len(corona_data)):
    corona_data_reshape[corona_data.index[i]] = corona_data.iloc[i].values

corona_data_reshape.index = corona_data.columns[:]

# adds total infected column and total infected sum by country
corona_data_reshape['Total Infected'] = corona_data_reshape.sum(axis=1)

# convert string to datetime
def parser(date):
    date = datetime.strptime(date,'%m/%d/%y')
    date  = str(date.day) + '-' + str(date.month) + '-' + str(date.year)
    return datetime.strptime(date,'%d-%m-%Y')

# convert str to datetime index
timestamp = []
for i in range(0, len(corona_data_reshape)):
    timestamp.append(parser(corona_data_reshape.index[i]))

corona_data_reshape.index = timestamp

# preparing for time series
infected_people = corona_data_reshape['Total Infected']

# add infected people per day column
difference = []
difference.append(corona_data_reshape['Total Infected'][0])

for i in range(0, len(corona_data_reshape['Total Infected']) - 1):
    difference.append(corona_data_reshape['Total Infected'][i+1] - corona_data_reshape['Total Infected'][i])

corona_data_reshape['Infected Per Day'] = difference    

# plot data
#plt.xlabel('Date')
#plt.ylabel('Infected People')
#infected_people.plot(figsize = (11, 5), marker='o')
#plt.legend()
#plt.show()
#print(infected_people.describe())

# trend plot
result = seasonal_decompose(infected_people)
#result.trend.plot(figsize=(12,4))
#plt.suptitle('COVID-19 Trend')
#plt.xlabel('Date')
#plt.ylabel('Infected People')
#plt.legend()
#plt.show()

# seasonality plot
#result.seasonal.plot(figsize=(12,4))
#plt.suptitle('COVID-19 Seasonality')
#plt.xlabel('Date')
#plt.ylabel('Infected People')
#plt.legend()
#plt.show()

# prediction plot
train = infected_people.iloc[:-8]
test = infected_people.iloc[-8:]

#prediction model
model = ExponentialSmoothing(train,trend = "mul", seasonal_periods = 7, seasonal = "add").fit()
prediction = model.predict(start = 65, end = 75)
plt.figure(figsize = (13, 5))

# plot prediction and actual
plt.plot(prediction, 'r', marker = 'o', markersize = 10, linestyle = '--', label='Predicted')
plt.plot(test, marker = 'o', markersize = 10, linestyle = '--', label='Actual')

# get labels, title, legend then show
plt.suptitle('COVID-19 Forecaster')
plt.xlabel('Date')
plt.ylabel('Infected People')
plt.legend()
print(prediction)
plt.show()