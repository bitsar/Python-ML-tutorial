import pandas
import quandl, math
import numpy
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression

# Variables and Switches
# Vars
predict = 0.1
NaN = -99999
# Switches
stock = "WIKI/GOOGL"
openPrice = 'Adj. Open'
closePrice = 'Adj. Close'
highPrice = 'Adj. High'
lowPrice = 'Adj. Low'
volumePrice = 'Adj. Volume'

# Data set initialisation
dataFrame = quandl.get(stock)
dataFrame = dataFrame[[openPrice, highPrice, lowPrice, closePrice, volumePrice]]
dataFrame['High-Low-Percent'] = (dataFrame[highPrice] - dataFrame[lowPrice]) / dataFrame[lowPrice] * 100.0
dataFrame['Percent-Change'] = (dataFrame[closePrice] - dataFrame[openPrice]) / dataFrame[openPrice] * 100.0
dataFrame = dataFrame[[closePrice, highPrice, 'Percent-Change', volumePrice]]
print(dataFrame.head())

# Learning Components
forecastColumn = closePrice

# Instruction to fill any NaN data in the dataframe with value=-99,999
dataFrame.fillna(value=NaN, inplace=True)

# Forecast ignore 1% of the entire length of the dataframe
# IE - if data is 100 days old, predict 1 day into future
forecastIgnore = int(math.ceil(predict * len(dataFrame)))
dataFrame['label'] = dataFrame[forecastColumn].shift(-forecastIgnore)

# Drop NaN data
dataFrame.dropna(inplace=True)

# Standard definitions
# Features
# Define numpy arrary of entire dataframe (except the label column)
X = numpy.array(dataFrame.drop(['label'], 1))
X = preprocessing.scale(X)

# Labels
# Define the label column of the dataframe
Y = numpy.array(dataFrame['label'])

# Training
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.2)

# Define classifiers (support vector regression)
# n_jobs=-1 : use all threads
classifier = LinearRegression(n_jobs=-1)
classifier.fit(X_train, Y_train)

# Testing
confidence = classifier.score(X_test, Y_test)
print(confidence)
