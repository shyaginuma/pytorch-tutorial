# Exercise 1: Basic Learning
# Please fill TODO

import torch
import torch.nn as nn

# Create tensors
x = torch.randn(100, 5)
y = torch.randn(100, 1)

# Build a fully connected layer.
linear = TODO
print("w: ", TODO)
print("b: ", TODO)

# Build loss function and optimizer.
criterion = TODO
optimizer = TODO

# Forward pass.
pred = TODO

# Compute loss.
loss = TODO
print("loss: ", TODO)

# Backward pass.
TODO

# Print out the gradients.
print("dL/dw: ", TODO)
print("dL/db: ", TODO)

# 1-step gradient descent.
TODO

# Print out the loss after 1-step gradient descent.
pred = TODO
loss = TODO
print("loss after 1 step optimization: ", TODO)
