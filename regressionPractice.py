from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
import numpy as np


# Use the make_regression function from scikit learn to create data for a regression.
# Using 1 feature so that we can easily graph the regression and verify that it's working properly.
X, y= make_regression(n_samples=200, n_features=1, noise=0.7, bias=27)

def linePlot(intercept, slope, X, y):
    max_x = np.max(X) + 200
    min_x = np.min(X) - 200

    xplot = np.linspace(min_x, max_x, 1000)
    yplot = intercept + slope * xplot

    plt.plot(xplot, yplot, color='#58b970', label='Regression Line')

    plt.scatter(X,y)
    plt.axis([-10, 10, 0, 200])
    plt.show()


# The h(x) function that returns the value for a current hypothesis.
def hypothesis(theta0, theta1, x):
    return theta0 + (theta1*x) 

# The cost function of a particular pair of (intercept, slope) values.
def cost(theta0, theta1, X, y):
    costValue = 0 
    for (xi, yi) in zip(X, y):
        costValue += 0.5 * ((hypothesis(theta0, theta1, xi) - yi)**2)
    return costValue

# The partial derivatives of the parameters so that we can minimize the cost.
def gradients(theta0, theta1, X, y):
    dtheta0 = 0
    dtheta1 = 0
    for (xi, yi) in zip(X, y):
        dtheta0 += hypothesis(theta0, theta1, xi) - yi
        dtheta1 += (hypothesis(theta0, theta1, xi) - yi)*xi

    dtheta0 /= len(X)
    dtheta1 /= len(X)

    return dtheta0, dtheta1

def updateParameters(theta0, theta1, X, y, alpha):
    dtheta0, dtheta1 = gradients(theta0, theta1, X, y)
    theta0 = theta0 - (alpha * dtheta0)
    theta1 = theta1 - (alpha * dtheta1)

    return theta0, theta1
    

def LinearRegression(X, y):
    theta0 = np.random.rand()
    theta1 = np.random.rand()
    
    for i in range(0, 1000):
        # print(cost(theta0, theta1, X, y))
        theta0, theta1 = updateParameters(theta0, theta1, X, y, 0.005)
    linePlot(theta0, theta1, X, y)


    


LinearRegression(X, y)