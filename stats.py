import numpy
from statistics import mean
import matplotlib.pyplot


# Define NumPy arrays and explicitly bind dataTypes
xs = numpy.array([1,2,3,4,5], dtype=numpy.float64)
ys = numpy.array([5,4,6,5,6], dtype=numpy.float64)

# Calculate regression
# Calculate mean of xs array and multiple with mean of ys array
# Devide by the square of xs array

def best_fit_slope(xs, ys):
    line_fit = (((mean(xs) * mean(ys)) - mean(xs*ys)) /
                ((mean(xs)**2) - mean(xs*xs)))
    return line_fit

line_fit = best_fit_slope(xs, ys)
print(line_fit)
