import math
import numpy
import quandl
import datetime
import matplotlib.pyplot
from matplotlib import style
from sklearn import preprocessing, model_selection
from sklearn.linear_model import LinearRegression

import variables as vars

# Data set initialisation
dataFrame = quandl.get(vars.stock)
dataFrame = dataFrame[[vars.openPrice, vars.highPrice, vars.lowPrice, vars.closePrice, vars.volumePrice]]
dataFrame['High-Low-Percent'] = (dataFrame[vars.highPrice] - dataFrame[vars.lowPrice]) / dataFrame[vars.lowPrice] * 100.0
dataFrame['Percent-Change'] = (dataFrame[vars.closePrice] - dataFrame[vars.openPrice]) / dataFrame[vars.openPrice] * 100.0
dataFrame = dataFrame[[vars.closePrice, vars.highPrice, 'Percent-Change', vars.volumePrice]]
print(dataFrame.head())

# Learning Components
forecastColumn = vars.closePrice

# Instruction to fill any NaN data in the dataframe with value=-99,999
dataFrame.fillna(value=vars.NaN, inplace=True)

# Forecast ignore 1% of the entire length of the dataframe
# IE - if data is 100 days old, predict 1 day into future
forecastIgnore = int(math.ceil(vars.predict * len(dataFrame)))
dataFrame['label'] = dataFrame[forecastColumn].shift(-forecastIgnore)

# Drop NaN data
dataFrame.dropna(inplace=True)

# Standard definitions
# Features
# Define numpy arrary of entire dataframe (except the label column)
X = numpy.array(dataFrame.drop(['label'], 1))
X = preprocessing.scale(X)
X_lately = X[-forecastIgnore:]
# X = X[:-forecastIgnore]
dataFrame.dropna(inplace=True)

# Labels
# Define the label column of the dataframe
Y = numpy.array(dataFrame['label'])

# Training
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.2)

# Classifiers
# n_jobs=-1 (use all threads)
classifier = LinearRegression(n_jobs=-1)
classifier.fit(X_train, Y_train)
confidence = classifier.score(X_test, Y_test)
print('\nKernel: Linear Regression ' + '\nConfidence: ', confidence)

# Forecasting
forecastSet = classifier.predict(X_lately)
print(forecastSet, confidence, forecastIgnore)