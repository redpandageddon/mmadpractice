import numpy as np

def compute_cost(X, y, theta):
    m = len(y)
    pred = X.dot(theta)
    cost = (1/ 2 * m) * np.sum(np.square(pred - y)) 
    
    return cost

def predict(X, theta):
    return np.dot(X, theta)

data = np.matrix(np.loadtxt('ex1data2.txt', delimiter=','))
X = data[:, 0:2]
X = np.c_[np.ones((X.shape[0], 1)), X]
y = data[:, 2]

a = np.linalg.pinv(np.dot(X.T, X))
b = a.dot(X.T)
c = b.dot(y)

print(c)

asd = compute_cost(X, y, c)
print(asd)

test = np.ones((2,3))
test[0][1] = 272000
test[1][1] = 314000
test[0][2] = 2
test[1][2] = 3

print('prediction ->' + str(predict(test, c)))