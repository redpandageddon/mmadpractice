import numpy as np
import math

def AbsArrayDifference(A, B):
    result = np.zeros((len(A)))
    
    for i in range(len(A)):
        result[i] = abs(A[i] - B[i])
        
    return result

def BestArray(arrayAmount, itemAmmount):
    arrays = np.zeros((arrayAmount, itemAmmount))
    sums = np.zeros((arrayAmount, 1))
    
    for i in range(arrayAmount):
        arrays[i] = 10 * np.random.random(itemAmmount) - 5
        sums[i] = np.sum(arrays[i])      
    
    index = np.argmax(sums)
    
    return arrays[index]

def EuclidDistance(A, B):
    diff = (A - B)**2
    result = math.sqrt(sum(diff.ravel()))
    
    return result