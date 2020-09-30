import numpy as np
import math

def AbsArrayDifference(A, B):
    result = np.zeros((len(A)))
    
    for i in range(len(A)):
        result[i] = abs(A[i] - B[i])
        
    return result

def BestArray(arrayAmount, itemAmmount):
    A = np.zeros((arrayAmount, itemAmmount))
    B = np.zeros((arrayAmount, 1))
    
    for i in range(arrayAmount):
        A[i] = 10 * np.random.random(itemAmmount) - 5
        B[i] = np.sum(A[i])      
    
    index = np.argmax(B)
    
    return A[index]

def EuclidDistance(A, B):
    diff = np.zeros((A.shape))
    diff = (A - B)**2
    asd = diff.ravel()
    result = math.sqrt(sum(asd))
    
    return result