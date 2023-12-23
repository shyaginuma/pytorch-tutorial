import numpy as np
import pandas as pd
import plotly.express as px
import torch
import torch.nn as nn

# Hyper-parameters
input_size = 1
output_size = 1
num_epochs = 60
learning_rate = 0.001

# Toy dataset
x_train = np.array(
    [
        [3.3],
        [4.4],
        [5.5],
        [6.71],
        [6.93],
        [4.168],
        [9.779],
        [6.182],
        [7.59],
        [2.167],
        [7.042],
        [10.791],
        [5.313],
        [7.997],
        [3.1],
    ],
    dtype=np.float32,
)

y_train = np.array(
    [
        [1.7],
        [2.76],
        [2.09],
        [3.19],
        [1.694],
        [1.573],
        [3.366],
        [2.596],
        [2.53],
        [1.221],
        [2.827],
        [3.465],
        [1.65],
        [2.904],
        [1.3],
    ],
    dtype=np.float32,
)

# Linear regression model
model = TODO

# Loss and optimizer
criterion = TODO
optimizer = TODO

# Train the model
for epoch in range(num_epochs):
    # Convert numpy arrays to torch tensors
    inputs = torch.from_numpy(x_train)
    targets = torch.from_numpy(y_train)

    # Forward pass
    outputs = TODO
    loss = TODO

    # Backward and optimize
    TODO

    if (epoch + 1) % 5 == 0:
        print("Epoch [{}/{}], Loss: {:.4f}".format(epoch + 1, num_epochs, loss.item()))

# Plot the graph
predicted = model(torch.from_numpy(x_train)).detach().numpy()
df = pd.DataFrame()
df["x"] = x_train
df["y"] = y_train
df["pred"] = predicted
df = pd.melt(df, id_vars=["x"], value_vars=["y", "y_pred"])
fig = px.scatter(
    df,
    x="x",
    y="value",
    color="variable",
)
fig.show()

# Save the model checkpoint
torch.save(model.state_dict(), "model.ckpt")
