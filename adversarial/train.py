import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms

import numpy as np

import matplotlib.pyplot as plt

from utils import train_model


# Load and transform our MNIST data
transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081)),
         transforms.Lambda(lambda x: x.repeat(3, 1, 1))]
    )
mnist_train = datasets.MNIST('./data', train=True, download=True,
                             transform=transform)
mnist_test = datasets.MNIST('./data', train=False, download=True,
                            transform=transform)

batch_size = 128

train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=False)

# Load model
model_ft = models.resnet18(pretrained=True)

# Replace last layer to proper amount of outputs for mnist
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 10)

model_ft = train_model(model_ft, train_loader, test_loader, epochs=25)
torch.save(model_ft.state_dict(), "weights/mnist.pth")
