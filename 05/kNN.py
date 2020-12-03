import numpy as np
import math

def k_nearest(X, k, obj):
    sub_X=X[:, :-1]
    distances=np.zeros(sub_X.shape[0])
    for i in range(sub_X.shape[0]):
        distances = [dist(item, obj) for item in sub_X]
        
    index = np.argsort(distances, axis = 0)
    nearest_classes=[X[index[item],2] for item in range(k)]
    
    unique, counts = np.unique(nearest_classes, return_counts=True)
    object_class = unique[np.argmax(counts)]
    return object_class

def dist(p1, p2):
    return math.sqrt(sum((p1 - p2)**2))





