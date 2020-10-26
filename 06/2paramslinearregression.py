import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
font = {'family': 'Verdana', 'weight': 'normal'}
rc('font', **font)

def normalize(arr):
    mean = np.mean(arr)
    arr = arr - mean
    std = np.std(arr)
    arr = arr / std
    
    return arr

def compute_cost(X, y, theta):
    m = len(y)
    pred = X.dot(theta)
    cost = (1/ 2 * m) * np.sum(np.square(pred - y)) 
    
    return cost

def predict(X, theta):
    return np.dot(X, theta)

def gradient_descent(X, y, alpha, iterations):
  m = X.shape[0]
  n = X.shape[1]
  theta = np.matrix('[0.0; 1.0; 1.0]')
  J_theta = np.zeros((iterations, 1))
  tmp = theta
  
  for i in range(iterations):
      pred = np.dot(X, tmp)
      J_theta[i] = compute_cost(X, y, tmp)
      r = ((1/m) * alpha * (X.T.dot((pred - y))))
      tmp = tmp - r
           
  plt.plot(np.arange(1000), J_theta, 'k-')
  plt.title('Снижение ошибки при градиентном спуске')
  plt.xlabel('Итерация')
  plt.ylabel('Ошибка')
  plt.grid()
  plt.show()
  
  return tmp

data = np.matrix(np.loadtxt('ex1data2.txt', delimiter=','))
square = data[:, 0]
room = data[:, 1]
price = data[:, 2]

normalize(square)
normalize(room)
normalize(price)

X = data[:, 0:2]
X_ones = np.c_[np.ones((X.shape[0], 1)), X]
y = price
theta = np.matrix('[1; 2; 3]')

primary_cost = compute_cost(X_ones, y, theta)
print('cost -> ' + str( primary_cost ))

theta = gradient_descent(X_ones, y, 0.00000001, 1000)
print('weights:')
print(theta)

new_cost = compute_cost(X_ones, y, theta)
print('cost -> ' + str( new_cost ))

print(primary_cost - new_cost)

test = np.ones((2,3))
test[0][1] = 272000
test[1][1] = 314000
test[0][2] = 2
test[1][2] = 3

print('prediction ->' + str(predict(test, theta)))


