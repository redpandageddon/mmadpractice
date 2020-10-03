import numpy as np

# евклидово расстояние между двумя точками
def dist(A, B, N):
  # TODO: реализуйте вычисление евклидова расстояния для двух точек A и B в N-мерном пространстве
  # Помните, что размерность пространства может быть произвольной (не только 2D).
  # Для взятия разностей по каждому измерерию можно использовать код: A - B
  # Для возведения в квадрат можете использовать оператор ** (например, 3**2 == 9) или функцию np.power
  # Для вычисления суммы используйте функцию sum или np.sum
  # Для вычисления квадратного корня используйте функцию np.sqrt
  # r = ...
  temp = 0
  
  for i in range(N):
      temp = ((A[i]-B[i])*(A[i]-B[i])) + temp

  r=np.sqrt(temp)
  return r


# возвращает список индексов ближайших центров по каждой точке
def class_of_each_point(X, centers):
  m = len(X)
  k = len(centers)

  # матрица расстояний от каждой точки до каждого центра
  distances = np.zeros((m, k))
  for i in range(m):
    for j in range(k):
      distances[i, j] = dist(centers[j], X[i], np.ndim(X))

  # поиск ближайшего центра для каждой точки
  return np.argmin(distances, axis=1)


def curse(k,X):
    m = X.shape[0]
    n = X.shape[1]

    curr_iteration = prev_iteration = np.zeros(m)
    centers = np.random.random((k,n))
    curr_iteration = class_of_each_point(X, centers)
    
    while True:

        prev_iteration = curr_iteration

    # вычисляем новые центры масс
        for i in range(k):
            sub_X = X[curr_iteration == i,:]
            if len(sub_X) > 0:
                centers[i,:] = np.mean(sub_X, axis=0)

    # приписываем каждую точку к заданному классу
        curr_iteration = class_of_each_point(X, centers)
    
        if np.all(curr_iteration==prev_iteration):
            break;
    
    return centers

def kmeans(k, X):

  # TODO: инициализировать переменные m и n
  # m - количество строк в матрице X
  # n - количество столбцов в матрице X
  # Используйте свойство shape матрицы X для решения этой задачи
  # Чтобы понять, что хранится в свойстве shape, попробуйте в консоли Python следующий код:
  # >>> ones = np.ones((3, 2))
  # >>> ones
  # >>> ones.shape
  # m = ...  # количество строк в матрице X
  # n = ...  # количество столбцов в матрице X
 

  # TODO: сгенерировать k кластерных центров со случайными координатами.
  # Должна получиться матрица случайных чисел размера k*n (k строк, n столбцов).
  # Для генерации матрицы случайных чисел используйте код:
  # centers = np.random.random((k, n))
  # Функция random генерирует случайные числа в диапазоне от 0 до 1, поэтому
  # не забывайте, что координаты центров не должны выходить
  # за границы минимальных и максимальных значений столбцов (признаков) матрицы X.
  # Для вычисления минимальных и максимальных значений по столбцам (признакам)
  # матрицы X используйте функции min(X, axis=0) и max(X, axis=0) библиотеки NumPy соответственно.
  # centers = ...

  ##############

  # приписываем каждую точку к заданному классу


  # цикл до тех пор, пока центры не стабилизируются
  # TODO: условие выхода из цикла - векторы curr_iteration и prev_iteration стали равны
  # Для сравнения двух массивов NumPy можете использовать один из вариантов:
  #   np.all(a1 == a2), где a1 и a2 массивы NumPy.
  # или
  #   np.any(a1 != a2)
  # Для реализации логического отрицания в Python используйте not
  # Поэкспериментируйте в консоли Python с функциями all и any, чтобы понять, как они работают.
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
