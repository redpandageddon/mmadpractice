import numpy as np

def AbsArrayDifference(A, B):
    A = np.asarray(A)
    B = np.asarray(B)
    if(A.shape != B.shape):
        raise

    return np.absolute(A - B)

def HighestSumArray(arrayAmount, itemAmmount):
    arrays = np.zeros((arrayAmount, itemAmmount))
    
    
    for i in range(arrayAmount):
        arrays[i] = 10 * np.random.random(itemAmmount) - 5      
    
    sums = np.sum(arrays, axis=1)
    index = np.argmax(sums)
    
    return arrays[index]

def EuclidDistance(A, B):
    diff = (A - B)**2

    return np.sqrt(sum(diff.ravel()))

def MinkovDistance(A, B, p, r):
    p = 2
    r = 2
    result = (A - B)
    
    return np.sum(abs(result.ravel()) ** p) ** (1/r)
