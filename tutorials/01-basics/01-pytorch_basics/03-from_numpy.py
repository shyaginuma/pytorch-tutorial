# Loading data from numpy
import numpy as np
import torch

# Create a numpy array.
x = np.array([[1, 2], [3, 4]])

# Convert the numpy array to a torch tensor.
y = torch.from_numpy(x)

# Convert the torch tensor to a numpy array.
z = y.numpy()

# check z
print("x: \n", x)
print("y: \n", y)
print("z: \n", z)
