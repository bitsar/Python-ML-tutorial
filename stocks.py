import pandas as pd
import quandl

# Data set initialisation
dataFrame = quandl.get("WIKI/GOOGL")
dataFrame = dataFrame[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
dataFrame['High-Low-Percent'] = (dataFrame['Adj. High'] - dataFrame['Adj. Low']) / dataFrame['Adj. Low'] * 100.0
dataFrame['Percent-Change'] = (dataFrame['Adj. Close'] - dataFrame['Adj. Open']) / dataFrame['Adj. Open'] * 100.0
dataFrame = dataFrame[['Adj. Close', 'High-Low-Percent', 'Percent-Change', 'Adj. Volume']]
print(dataFrame.head())