import numpy as np
import matplotlib.pyplot as plt
from fns import compute_cost, gradient_descent, predict
from matplotlib import rc
font = {'family': 'Verdana', 'weight': 'normal'}
rc('font', **font)

def normalize(arr):
    mean = np.mean(arr)
    arr = arr - mean
    std = np.std(arr)
    arr = arr / std
    
    return arr

# Initail data
data = np.matrix(np.loadtxt('ex1data2.txt', delimiter=','))
square = data[:, 0]
room = data[:, 1]
price = data[:, 2]

# Normalized data
square = normalize(square)
room = normalize(room)
price = normalize(price)


X = data[:, 0:2]
X_ones = np.c_[np.ones((X.shape[0], 1)), X]
y = price
theta = np.matrix('[1; 2; 3]')

# Computing initial cost
primary_cost = compute_cost(X_ones, y, theta)
print('initial cost -> ' + str( primary_cost ))

# Gradient descent
theta, J_th = gradient_descent(X_ones, y, 0.000000002, 1000)
plt.plot(np.arange(1000), J_th, 'k-')
plt.title('Снижение ошибки при градиентном спуске')
plt.xlabel('Итерация')
plt.ylabel('Ошибка')
plt.grid()
plt.show()

# New weights
print('weights:')
print(theta)

# New cost
new_cost = compute_cost(X_ones, y, theta)
print('new cost -> ' + str( new_cost ))

# Cost difference
print('cost  difference -> ' + str( primary_cost - new_cost ))

# Test
test = np.ones((2,3))
test[0][1] = 272000
test[1][1] = 314000
test[0][2] = 2
test[1][2] = 3
print('prediction ->' + str(predict(test, theta)))


