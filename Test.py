import numpy as np

x = np.array([-9,2, 3,-4, -5, 6])
y = np.argpartition(x, -3)[:3]
print x[y]