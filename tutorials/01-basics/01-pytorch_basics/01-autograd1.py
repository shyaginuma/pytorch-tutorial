# Basic autograd example 1

import torch

# Create tensors.
x = torch.tensor(1.0, requires_grad=True)
w = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(3.0, requires_grad=True)

# Build a computational graph.
y = w * x + b  # y = 2 * x + 3

# Compute gradients.
y.backward()

# Print out the gradients.
print(x.grad)  # x.grad = 2
print(w.grad)  # w.grad = 1
print(b.grad)  # b.grad = 1
