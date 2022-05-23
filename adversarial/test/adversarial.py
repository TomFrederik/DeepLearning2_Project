import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import torchvision
from torchvision import datasets, models, transforms

import torchattacks
from torchattacks.attack import Attack
from torchattacks import PGD, FGSM

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import utils
from typing import Any, Tuple, List, Iterable


def get_attack(
    attack_name: str,
    model: nn.Module, **kwargs: Any
) -> Attack:

    attack_dict = {
        "pgd": torchattacks.PGD(model, **kwargs),
        "fgsm": torchattacks.FGSM(model, **kwargs)
    }

    return attack_dict[attack_name]


def evaluate_model_attack(
    attack_name: str, model: nn.Module, dataset: Dataset[Any], sample_size: int
) -> Tuple[List[float], List[float], np.ndarray]:

    accuracies = []
    adv_accuracies = []
    eps_range = []

    avg_adv_accuracy = np.inf

    eps_range = np.linspace(0, 1, 20)

    pbar = tqdm(eps_range)

    for eps in pbar:

        loader = DataLoader(dataset, batch_size=sample_size, shuffle=True)
        inputs, labels = next(iter(loader))

        atk = get_attack(attack_name, model, eps=eps)
        atk = torchattacks.PGD(model, alpha=0.1, eps=eps, steps=10)

        correct = 0
        adv_correct = 0
        total = 0

        outputs = model(inputs)

        adv_images = atk(inputs, labels)
        adv_outputs = model(adv_images)

        _, preds = torch.max(outputs, 1)
        _, adv_preds = torch.max(adv_outputs, 1)

        correct = (preds == labels).sum()
        adv_correct = (adv_preds == labels).sum()

        accuracy = (100 * float(correct) / sample_size)
        adv_accuracy = (100 * float(adv_correct) / sample_size)

        accuracies.append(accuracy)
        adv_accuracies.append(adv_accuracy)

        pbar.set_description(f"acc: {np.round(adv_accuracy, 2)} - eps: {np.round(eps, 3)}")

    return accuracies, adv_accuracies, eps_range


sample_size = 256

# Load model and load trained weights from disk
model = models.resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10)

model.load_state_dict(torch.load("weights/mnist.pth"))
model.eval()

# Load validation data and transform it in same manner as during training
transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081)),
         transforms.Lambda(lambda x: x.repeat(3, 1, 1))]
    )

mnist_test = datasets.MNIST('./data', train=False, download=True,
                            transform=transform)

attack_name = "pgd"
accuracies, adv_accuracies, eps_range = evaluate_model_attack(attack_name, model, mnist_test, sample_size)

plt.title(f"Attack: {attack_name}\nAccuracy on original dataset: {np.round(np.mean(accuracies), 2)}")
plt.plot(eps_range, adv_accuracies, label="Adversarial accuracy")
plt.xlabel("eps")
plt.ylabel("accuracy")
plt.ylim([0, 100])
plt.show()
