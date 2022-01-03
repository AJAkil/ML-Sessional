import numpy as np

a = np.array([
    [1,2,3],
    [1,2,3]
])

b = np.array([1,2,3])
print(b.shape)
print(np.sum(a, axis=1))

result = np.sum((a * b), axis=1) / np.sum(a, axis=1)
print(result)
