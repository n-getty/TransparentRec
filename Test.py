import numpy as np


x = np.array([1,10,11,10])
y = np.array([3,4,5,6])
p = np.where(x>2)
x[p] = 0
print x[p]
print x