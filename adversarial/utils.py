import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms

import numpy as np

import copy
import sys
import time
from tqdm import tqdm

import matplotlib.pyplot as plt


def train_model(model, train_loader, test_loader, epochs=25):
    since = time.time()

    device = torch.device("cuda") if torch.cuda.is_available else torch.device("cpu")
    model = model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    best_model_weights = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    train_n = len(train_loader) * train_loader.batch_size
    test_n = len(test_loader) * test_loader.batch_size

    for epoch in range(epochs):

        print(f"Epoch {epoch + 1}/{epochs}")
        print("-" * 10)

        running_loss = 0.0
        running_corrects = 0

        model.train()
        for batch in tqdm(train_loader, desc="train"):
            inputs, labels = batch

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = loss_fn(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item() * inputs.shape[0]
            running_corrects += torch.sum(preds == labels)

        train_epoch_loss = running_loss / train_n
        train_epoch_acc = running_corrects.double() / train_n

        print(f"Train Loss: {train_epoch_loss:.4f} Acc: {train_epoch_acc:.4f}")

        model.eval()

        running_loss = 0.0
        running_corrects = 0

        for batch in tqdm(test_loader, desc="test"):
            inputs, labels = batch

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = loss_fn(outputs, labels)

            running_loss += loss.item() * inputs.shape[0]
            running_corrects += torch.sum(preds == labels)

        val_epoch_loss = running_loss / test_n
        val_epoch_acc = running_corrects.double() / test_n

        if val_epoch_acc > best_acc:
            best_acc = val_epoch_acc
            best_model_weights = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), "weights/mnist_during.pth")

        print(f"Val Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f}")

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_weights)
    utils.visualize_preds(model, test_loader)

    return model


def visualize_preds(model, val_loader):
    model.eval()
    images_so_far = 0
    device = torch.device("cuda") if torch.cuda.is_available else torch.device("cpu")

    num_images = 16

    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(val_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.shape[0]):
                images_so_far += 1
                ax = plt.subplot(num_images // 2, 2, images_so_far)
                ax.axis("off")
                ax.set_title(f"predicted: {preds[j]}")
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    return


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.1307, 0.1307, 0.1307])
    std = np.array([0.3081, 0.3081, 0.3081])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated
