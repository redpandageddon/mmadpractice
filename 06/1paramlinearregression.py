import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from fns import compute_cost, gradient_descent, predict
from matplotlib import rc
font = {'family': 'Verdana', 'weight': 'normal'}
rc('font', **font)

# Initial data
data = np.matrix(np.loadtxt('ex1data1.txt', delimiter=','))
X = data[:, 0]
y = data[:, 1]
plt.plot(X, y, 'g.')
plt.title('Зависимость прибыльности от численности')
plt.xlabel('Численность')
plt.ylabel('Прибыльность')
plt.show()

# Computing cost b4 gradient descent
m = X.shape[0]
X_ones = np.c_[np.ones((m, 1)), X]
theta = np.matrix('[1; 2]')
reg = LinearRegression().fit(X, y)
print(compute_cost(X_ones, y, theta))

# Gradient descent
theta, J_th = gradient_descent(X_ones, y, 0.02, 500)
print(theta)

# Cost change while gradient descent 
plt.plot(np.arange(500), J_th, 'k-')
plt.title('Снижение ошибки при градиентном спуске')
plt.xlabel('Итерация')
plt.ylabel('Ошибка')
plt.grid()
plt.show()

# Test after grandient descent
test = np.ones((2,2))
test[0][1] = 2.72
test[1][1] = 3.14
test_prediction = predict(test, theta)
print('selfmade predictions: ' + str (test_prediction))
print('module predictions: ')
check = reg.predict(np.array([[2.72]]))
print(check)
check = reg.predict(np.array([[3.14]]))
print(check)

# Regression line
x = np.arange(min(X), max(X))
plt.plot(x, theta[1] * x.ravel() + theta[0], 'g--')
plt.plot(X, y, 'b.')
plt.grid()
plt.show()

check = reg.predict(X)
plt.scatter(np.asarray(X), np.asarray(y),  color='black')
plt.plot(X, check, color='blue', linewidth=3)
plt.xticks(())
plt.yticks(())
plt.show()
