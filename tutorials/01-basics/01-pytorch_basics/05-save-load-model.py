# Save and load the model

import torch
import torchvision

# Load pretrained model
resnet = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)

# Save and load the entire model.
torch.save(resnet, "model.ckpt")
model = torch.load("model.ckpt")

# Save and load only the model parameters (recommended).
torch.save(resnet.state_dict(), "params.ckpt")
resnet.load_state_dict(torch.load("params.ckpt"))
