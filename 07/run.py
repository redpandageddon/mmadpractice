import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from displayData import displayData
from predict import predict

test_set = sio.loadmat('test_set.mat')
th_set = sio.loadmat('weights.mat')

x_axis = test_set['X']
y_axis = np.int64(test_set['y'])
th1 = th_set['Theta1']
th2 = th_set['Theta2']

m = x_axis.shape[0]
indexes = np.random.permutation(m)
x = np.zeros((100,x_axis.shape[1]))

for i in range(100):
    x[i] = x_axis[indexes[i]]
    
displayData(x)

pre = predict(x_axis, th1, th2)
y_axis.ravel()

q = (pre == y_axis.ravel())
print(q)
res = np.mean(np.double(q))
print('Accurasy: ' + str(res * 100) + '%%')

rp = np.random.permutation(m)
plt.figure()
for i in range(5):
     X2 = x_axis[rp[i],:]
     X2 = np.matrix(x_axis[rp[i]])
     pred = predict(X2.getA(), th1, th2)
     pred = np.squeeze(pred)
     pred_str = 'Neural Network Prediction: %d (digit %d)' % (pred, y_axis[rp[i]])
     displayData(X2, pred_str)
     plt.close()
     
mistake = np.where(pre != y_axis.ravel())[0]
qwerty = np.zeros((100,x_axis.shape[1]))
for i in range(100):
    qwerty[i] = x_axis[mistake[i]]
displayData(qwerty)