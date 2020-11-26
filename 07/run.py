import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from displayData import displayData
from predict import predict

test_set = sio.loadmat('test_set.mat')
th_set = sio.loadmat('weights.mat')

X_test = test_set['X']
y_test = np.int64(test_set['y'])
th1 = th_set['Theta1']
th2 = th_set['Theta2']

m = X_test.shape[0]

def random(m):
   return np.random.permutation(m)

indexes = random(m)
x = np.zeros((100,X_test.shape[1]))

for i in range(100):
    x[i] = X_test[indexes[i]]
    
displayData(x)

pre = predict(X_test, th1, th2)
y_test.ravel()

accuracy = np.mean(np.double(pre == y_test.ravel()))
print('Accuracy: ' + str(accuracy * 100) + '%%')

rp = random(m)
plt.figure()
for i in range(5):
     X2 = X_test[rp[i],:]
     X2 = np.matrix(X_test[rp[i]])
     pred = predict(X2.getA(), th1, th2)
     pred = np.squeeze(pred)
     pred_str = 'Neural Network Prediction: %d (digit %d)' % (pred, y_test[rp[i]])
     displayData(X2, pred_str)
     plt.close()
     
mistake = np.where(pre != y_test.ravel())[0]
wrongly_predicted = np.zeros((100,X_test.shape[1]))
for i in range(100):
    wrongly_predicted[i] = X_test[mistake[i]]
displayData(wrongly_predicted)
