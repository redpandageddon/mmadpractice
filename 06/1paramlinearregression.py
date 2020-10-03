import numpy as np
import matplotlib.pyplot as plt

def compute_cost(X, y, theta):

    h_x = X * theta
    cost = np.sum(np.power(h_x - y, 2)) / (2 * X.shape[0])
    
    return cost

def gradient_descent(X, y, alpha, iterations):
    
  m = X.shape[0]
  n = X.shape[1]
  theta = np.ones((n, 1))
  theta[0] = 0
  J_theta = np.zeros((iterations, 1))
  tmp = theta
  
  for i in range(iterations):
      J_theta[i] = compute_cost(X, y, tmp)
      k = (alpha / m)
      s = np.zeros((n, 1))
      for j in range(0, n):
          s[j] = (np.multiply((X * tmp - y), X[:, j])).sum()
          tmp[j] = tmp[j] - k * s[j]
           
  plt.plot(np.arange(500), J_theta, 'k-')
  plt.title('Снижение ошибки при градиентном спуске')
  plt.xlabel('Итерация')
  plt.ylabel('Ошибка')
  plt.grid()
  plt.show()
  return tmp

def predict(X, theta):
    return np.dot(X, theta)


from matplotlib import rc
font = {'family': 'Verdana', 'weight': 'normal'}
rc('font', **font)

data = np.matrix(np.loadtxt('ex1data1.txt', delimiter=','))
X = data[:, 0]
y = data[:, 1]
plt.plot(X, y, 'g.')
plt.title('Зависимость прибыльности от численности')
plt.xlabel('Численность')
plt.ylabel('Прибыльность')
plt.show()

m = X.shape[0]
X_ones = np.c_[np.ones((m, 1)), X]
theta = np.matrix('[1; 2]')
print(compute_cost(X_ones, y, theta))

theta = gradient_descent(X_ones, y, 0.02, 500)
print(theta)

test = np.ones((2,2))
test[0][1] = 2.72
test[1][1] = 3.14
asd = predict(test, theta)
print(asd)

x = np.arange(min(X), max(X))
plt.plot(x, theta[1]*x.ravel() + theta[0], 'g--')
plt.plot(X, y, 'b.')
plt.grid()
plt.show()
