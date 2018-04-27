import pandas as pd
import quandl, math
import numpy
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression

# Data set initialisation
dataFrame = quandl.get("WIKI/GOOGL")
dataFrame = dataFrame[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
dataFrame['High-Low-Percent'] = (dataFrame['Adj. High'] - dataFrame['Adj. Low']) / dataFrame['Adj. Low'] * 100.0
dataFrame['Percent-Change'] = (dataFrame['Adj. Close'] - dataFrame['Adj. Open']) / dataFrame['Adj. Open'] * 100.0
dataFrame = dataFrame[['Adj. Close', 'High-Low-Percent', 'Percent-Change', 'Adj. Volume']]
print(dataFrame.head())