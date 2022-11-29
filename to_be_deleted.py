import numpy as np


a = np.array([1, 2, 3, 4])

print('a', a)

b = np.array([a, a])

print('b', b)

c = b.flatten('F')

print('c', c)