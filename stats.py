import numpy
from statistics import mean
import matplotlib.pyplot
from matplotlib import style
style.use('ggplot')


# Define NumPy arrays and explicitly bind dataTypes
xs = numpy.array([1,2,3,4,5], dtype=numpy.float64)
ys = numpy.array([5,4,6,5,6], dtype=numpy.float64)

# Calculate regression
# Calculate mean of xs array and multiple with mean of ys array
# Devide by the square of xs array
def best_fit_slope_and_intercept(xs, ys):
    gradient = (((mean(xs) * mean(ys)) - mean(xs*ys)) /
                ((mean(xs)**2) - mean(xs*xs)))

    best_fit = mean(ys) - gradient*mean(xs)

    return gradient, best_fit

gradient, best_fit = best_fit_slope_and_intercept(xs, ys)
print(gradient, best_fit)

# Calculate regression line
regression_line =[]
for x in xs:
    regression_line.append((gradient*x) + best_fit)

# Predictions
predict_x = 7
predict_y = (gradient*predict_x) + best_fit
print(predict_y)

# Calculate error margins
# Squared error
def squared_error(ys_original, ys_line):
    return sum(ys_line - ys_original) * (ys_line - ys_original)

# Graph output
matplotlib.pyplot.scatter(xs, ys, color='#003F72', label='data')
matplotlib.pyplot.plot(xs, regression_line, label='regression line')
matplotlib.pyplot.legend(loc=4)
matplotlib.pyplot.show()

