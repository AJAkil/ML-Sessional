import numpy as np

a = np.array([[1], [2], [3]])
b = np.vstack(
    (
        a.T,
        a.T
    )
)

print(b)
print(b.shape)
