import matplotlib.pyplot as plt
import autograd.numpy as np
import pandas as pd
from autograd import grad

# Access the datasets and name the columns
training_data = pd.read_csv("data/training_data.csv", header = None)
validation_data = pd.read_csv("data/validation_data.csv", header = None)
column_names = ["x_values", "y_values"]
validation_data.columns = column_names
training_data.columns = column_names
# Initializing the regressive function
parameters = np.array([10.0, -10.0, 10.0, 0.0]) # parameter [a, b, c, d]


learning_rate = 1e-18
MSL_progression = []



def f(a, b, c, d, x):
    return b*np.sin(a*x**c) + d


# defining the calculate loss function (mean squared loss = MSL)
def L(a, b, c, d, x, true_y):
    return (f(a, b, c, d, x) - true_y)**2


# calculating the MSL for the current parameters
def calculate_MSL(data, parameters):
    losses = []
    for i in range(0, len(data)):
        x = float(data.iloc[i, 0])
        true_y = data.iloc[i, 1]
        losses.append(L(parameters[0], parameters[1], parameters[2], parameters[3], x, true_y))
    return sum(losses) / len(losses)


def Grad_descent(data):
    for i in range(len(parameters)):
        dLdn = grad(L, argnum=i)
        sum_derivative = 0
        for k in range(len(data)):
            x = np.array(data.iloc[k, 0])
            true_y = np.array(data.iloc[k, 1])
            sum_derivative += dLdn(parameters[0], parameters[1], parameters[2], parameters[3], x, true_y)
        mean_derivative = sum_derivative / len(data)
        parameters[i] -= learning_rate * mean_derivative

def rep(data):
    Grad_descent(data)
    return calculate_MSL(data, parameters)


for i in range(0,100):
    print(rep(training_data))
    print(parameters)


pred_x = np.array(range(1, 361))
pred_y = parameters[1]*np.sin(parameters[0]*pred_x**parameters[2]) + parameters[3]

plt.scatter(pred_x, pred_y, c = "blue")
plt.scatter(training_data.iloc[:, 0], training_data.iloc[:, 1], c = "black")
plt.show()





#def g(x, y):
#    return x**2 + 2*y**2
#dgdx = grad(g, argnum = 0)
#x_val = np.array(1.0)
#y_val = np.array(1.0)
#print(dgdx(x_val, y_val))