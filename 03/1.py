import numpy as np
import matplotlib.pyplot as plt
import scipy.io


def autocorrelate(a):
  n = len(a)
  cor = []
  for i in range(n//2, n//2+n):
    a1 = a[:i+1]   if i< n else a[i-n+1:]
    a2 = a[n-i-1:] if i< n else a[:2*n-i-1]
    cor.append(np.corrcoef(a1, a2)[0, 1])
  return np.array(cor)

a = 10
# переменная
print(a)

a = [1,2,3]
# массив
print(a)

a = [[1],[1,2],[1,2,3]]
# матрица с заданными значениями
print(a)

a = np.zeros((2,3))
# матрица с нулевыми значениями
print(a)

a = np.ones((2,3))
# матрица с единичными значениями
print(a)

a = np.random.randint(2, 6, (5, 5))
# матрица со случайными целочисленными значениями
print(a)

data = np.random.normal(3, 5, 1000)
print(data)

data = np.loadtxt('./test.txt', dtype=np.int32)
print(data)

data = scipy.io.loadmat('./1D/var1.mat')
data.keys()
data=data['n']

dataMax=np.max(data)
print(dataMax)

dataMin=np.min(data)
print(dataMin)

dataMedian=np.median(data)
print(dataMedian)

dataMean=np.mean(data)
print(dataMean)

dataVar=np.var(data)
print(dataVar)

dataStd=np.std(data)
print(dataStd)

plt.plot(data)
plt.show()

mean = np.mean(data) * np.ones(len(data))
var = np.var(data) * np.ones(len(data))
plt.plot(data, 'b-', mean, 'r-', mean-var, 'g--', mean+var, 'g--')
plt.grid()
plt.show()

plt.hist(data, bins=20)
plt.grid()
plt.show()

data = np.ravel(data)
cor = autocorrelate(data)
plt.plot(cor)
plt.show()

datadata = scipy.io.loadmat('./ND/var1.mat')
datadata.keys()
datadata=datadata['mn']

n = datadata.shape[1]
corr_matrix = np.zeros((n,n))

for i in range(0,n):
    for j in range(0,n):
        a1 = datadata[:,i]
        a2 = datadata[:,j]
        corr_matrix[i,j]=np.corrcoef(a1,a2)[0,1]

np.set_printoptions(precision=2)
print(corr_matrix)

plt.plot(datadata[:, 2], datadata[:, 5], 'b.')
plt.grid()
plt.show()
