import numpy as np
import math

def normalize(arr): 
    mean = np.mean(arr, axis=0) 
    arr = arr - mean 
    std = np.std(arr, axis=0) 
    arr = arr / std 
    return arr

def k_nearest(X, k, obj):
    sub_X = normalize(X[:, :-1])
    distances = [dist(item, obj) for item in sub_X]
        
    index = np.argsort(distances, axis = 0)
    nearest_classes = X[index[:k], 2]
    
    unique, counts = np.unique(nearest_classes, return_counts=True)
    object_class = unique[np.argmax(counts)]
    return object_class

def dist(p1, p2):
    return math.sqrt(sum((p1 - p2)**2))
