import numpy as np

def dist(A, B, N):
  temp = 0
  
  for i in range(N):
      temp = ((A[i]-B[i])*(A[i]-B[i])) + temp

  r = np.sqrt(temp)
  return r


def class_of_each_point(X, centers):
  m = len(X)
  k = len(centers)

  distances = np.zeros((m, k))
  for i in range(m):
    for j in range(k):
      distances[i, j] = dist(centers[j], X[i], np.ndim(X))

  return np.argmin(distances, axis=1)


def curse(k,X):
    m = X.shape[0]
    n = X.shape[1]

    curr_iteration = prev_iteration = np.zeros(m)
    centers = np.random.random((k,n))
    curr_iteration = class_of_each_point(X, centers)
    
    while (curr_iteration!=prev_iteration):

        prev_iteration = curr_iteration

        for i in range(k):
            sub_X = X[curr_iteration == i,:]
            if len(sub_X) > 0:
                centers[i,:] = np.mean(sub_X, axis=0)

        curr_iteration = class_of_each_point(X, centers)
    
    return centers

def kmeans(k, X):
  while True:
     centers = curse(k,X)
     if check(X,centers)==True:
         break;
  return centers

def check(X, centers):
    for i in range(centers.shape[0]):
        for j in range(centers.shape[1]):
            if (np.min(X[:,j], axis=0)>centers[i,j]) or (np.max(X[:,j],axis=0)<centers[i,j]):
                return False
    return True
