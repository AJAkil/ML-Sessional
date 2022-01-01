import numpy as np

a = np.array([1, 2])
b = np.array([1, 2])

print((a + b) * 0.22)

a = a.reshape(a.shape[0], 1)
b = b.reshape(b.shape[0], 1)
print((a + b) * 0.22)
