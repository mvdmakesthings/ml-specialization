from matplotlib import pyplot as plt
import numpy as np

# Plot the regression line
def plotRegressionLine(x, y, degree, color, label, ax):
    coefficient = np.polyfit(x, y, deg=degree)
    f_wb = np.poly1d(coefficient)
    xx = np.linspace(0,9,9)
    yy = f_wb(xx)
    line, = ax.plot(xx, yy, c=color, label=label)
    return line, yy

# Calculate Cost of function
def squaredErrorCost(x, y, y_hat):
    cost = 0
    diff = np.sum(y_hat - y)**2
    cost = (1/(2*len(x))) * diff
    return cost

# Plots a graph as a subplot
def plotAx(ax, data, title, dimension):
    x,y = data
    ax.set_title(title)
    ax.set(xlim=(0, 10), ylim=(0, 1000))
    ax.scatter(x, y, c="black")
    line, y_hat = plotRegressionLine(x, y, degree=dimension, color="orange", label="Regression Line", ax=ax)
    cost = squaredErrorCost(x, y, y_hat)
    ax.text(1, 700, f'Cost: {cost:.0f}', style='italic', bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 3})

# Data Normalization

# Standard normalization function
def normalize_std(x):
    max = np.max(x)
    return x / max

# Mean normalization function
def normalize_mean(x):
    max = np.max(x)
    min = np.min(x)
    mu = np.mean(x)
    norm = (x-mu)/(max-min)
    return norm

# Z-Score normalization function
def normalize_zscore(x):
    mean = np.mean(x)
    std_deviation = np.std(x)
    return (x - mean) / std_deviation