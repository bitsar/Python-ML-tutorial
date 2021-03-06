import math
import numpy
import quandl
import datetime

# Maths and Science libraries/modules
import matplotlib.pyplot
import pickle
from matplotlib import style
from sklearn import preprocessing, model_selection
from sklearn.linear_model import LinearRegression


# Local imports
import variables as vars

# Style set
style.use('ggplot')

# DataFrame initialisations
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
# dataFrame.dropna(inplace=True)

# Standard definitions
# Features
# Define numpy arrary of entire dataframe (except the label column)
X = numpy.array(dataFrame.drop(['label'], 1))
X = preprocessing.scale(X)
X_lately = X[-forecastIgnore:]
X = X[:-forecastIgnore]
dataFrame.dropna(inplace=True)

# Labels
# Define the label column of the dataframe
Y = numpy.array(dataFrame['label'])

# Training Components
# [Classifiers, Forecasting and Slicing]
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.2)

# Classifiers
# n_jobs=-1 (use all threads)
if vars.classifierSwitch == 'on':
    classifier = LinearRegression(n_jobs=-1)
    classifier.fit(X_train, Y_train)
    confidence = classifier.score(X_test, Y_test)
    print('\nKernel: Linear Regression ' + '\nConfidence: ', confidence)

    # Use pickle module to save classifier as Python object.
    # Use file_handler to (w)rite (b)inary to *.pickle object (as file)
    with open('linearRegression.pickle', 'wb') as pickle_out:
        pickle.dump(classifier, pickle_out)

# Rehydrate pickle
# Use file_handler to (r)ead (b)inary *.pickle and load back into classifier (training)
pickle_in = open('linearRegression.pickle', 'rb')
classifier = pickle.load(pickle_in)


# Forecasting
forecastSet = classifier.predict(X_lately)
dataFrame['Forecast'] = numpy.nan
if vars.classifierSwitch == 'on':
    print(forecastSet, confidence, forecastIgnore)

# Slicing and Dicing
lastDate = dataFrame.iloc[-1].name
lastUnix = lastDate.timestamp()
oneDay = vars.oneDay
nextUnix = lastUnix + oneDay

# Iterate through the forecast set
# Set daily values into dataframe (making all value != NaN)
for i in forecastSet:
    nextDate = datetime.datetime.fromtimestamp(nextUnix)
    nextUnix += vars.oneDay
    dataFrame.loc[nextDate] = [numpy.nan for _ in range (len(dataFrame.columns)-1)] + [i]

# Graph output
dataFrame[vars.closePrice].plot()
dataFrame['Forecast'].plot()
matplotlib.pyplot.legend(loc=4)
matplotlib.pyplot.xlabel('Date')
matplotlib.pyplot.ylabel('Price')
matplotlib.pyplot.show()

