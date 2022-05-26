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

import os
from os.path import join

import adv_utils
from typing import Any, Tuple, List, Iterable
import wandb

class AvgPredGetter(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        return out["avg_preds"]


def get_attack(
    attack_name: str,
    model: nn.Module,
) -> Attack:

    attack_dict = {
        "pgd": torchattacks.PGD,
        "fgsm": torchattacks.FGSM
    }

    return attack_dict[attack_name]


def evaluate_model_attack(
    attack_name: str, model: nn.Module,
    dataloader: DataLoader[Any], **kwargs: Any
) -> Tuple[List[float], List[float], np.ndarray]:

    device = torch.device("cuda") if torch.cuda.is_available else torch.device("cpu")

    accuracies = []
    adv_accuracies = []
    eps_range = []

    eps_range = np.linspace(0, 1, 5)
    # eps_range = np.linspace(0, 1, 5)

    pbar = tqdm(eps_range, desc="eps")

    for eps in pbar:

        correct = 0
        adv_correct = 0
        total = 0

        atk = get_attack(attack_name, model)
        atk = atk(model, eps=eps, **kwargs)

        for _ in tqdm(range(16), desc="batches"):
            batch = next(iter(dataloader))

            images, labels = batch["ims"], batch["labels"]
            images, labels = images.to(device), labels.to(device)
            total += len(labels)

            outputs = model(images)

            adv_images = atk(images, labels)

            adv_outputs = model(adv_images)

            _, preds = torch.max(outputs, 1)
            _, adv_preds = torch.max(adv_outputs, 1)

            correct += ((preds == labels).sum()).cpu()
            adv_correct += ((adv_preds == labels).sum()).cpu()

        accuracy = correct / total
        adv_accuracy = adv_correct / total

        accuracies.append(accuracy)
        adv_accuracies.append(adv_accuracy)

    return accuracies, adv_accuracies, eps_range


if __name__ == "__main__":

    os.makedirs("adversarial", exist_ok=True)

    classifier_name = "classifier_2022_05_25_16_39_indist"

    path = join("imagenette", "experiments", classifier_name)

    model, args = adv_utils.load_model(path)

    device = torch.device("cuda") if torch.cuda.is_available else torch.device("cpu")

    model = model.to(device)

    model = AvgPredGetter(model)

    args["batch_size"] = 16
    train_loader, val_loader, train_sampler = adv_utils.get_dataloaders(args)

    pgd_accuracies, pgd_adv_accuracies, eps_range = evaluate_model_attack("pgd", model, val_loader, steps=50)
    fgsm_accuracies, fgsm_adv_accuracies, eps_range = evaluate_model_attack("fgsm", model, val_loader)

    wandb.init(project="DL2", entity="xinyichen", config = args, name ="adversarial" +"_"+str(eps_range))

    fig = plt.figure()
    plt.title(f"PGD and FGSM accuracy")
    plt.plot(eps_range, pgd_adv_accuracies, label=f"PGD Adv. Acc. (orig. acc: {np.mean(pgd_accuracies):.3f})")
    plt.plot(eps_range, fgsm_adv_accuracies, label=f"FGSM Adv. Acc. (orig. acc: {np.mean(fgsm_accuracies):.3f})")
    plt.xlabel("eps")
    plt.ylabel("accuracy")
    plt.ylim([0, 1])
    plt.legend()
    plt.show()
    plt.savefig(f"adv_{classifier_name}_{str(eps_range)}.jpg")

    wandb.Image(fig)

    data = [[x, y] for (x, y) in zip(eps_range, pgd_adv_accuracies)]
    table = wandb.Table(data=data, columns = ["eps", "accuracy"])
    wandb.log({f"PGD Adv. Acc. (orig. acc: {np.mean(pgd_accuracies):.3f})": 
                wandb.plot.line(table, "eps", "accuracy",
                title="PGD  accuracy")})

    data = [[x, y] for (x, y) in zip(eps_range, fgsm_adv_accuracies)]
    table = wandb.Table(data=data, columns = ["eps", "accuracy"])
    wandb.log({f"FGSM Adv. Acc. (orig. acc: {np.mean(fgsm_accuracies):.3f})": 
                wandb.plot.line(table, "eps", "accuracy",
                title="FGSM accuracy")})
