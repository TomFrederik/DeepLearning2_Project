from numpy.lib import utils
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
from copy import deepcopy

import os
from os.path import join
import argparse

import adv_utils
from typing import Any, Tuple, List, Iterable, Dict


class ResNet50Model(nn.Module):
    """
    Wrapper around ResNet50 model that only returns
    the values for the 10 classes we care about
    """
    def __init__(self, device: torch.device) -> None:
        super().__init__()
        self.model = models.resnet50(pretrained=True).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)  # type: torch.Tensor
        return out


class AvgPredGetter(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)  # type: Dict[str, torch.Tensor]
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
    dataloader: DataLoader[Any],
    eps_steps: int = 5,
    **kwargs: Any
) -> Tuple[List[float], List[float], np.ndarray]:

    device = torch.device("cuda") if torch.cuda.is_available else torch.device("cpu")

    # Create folder for intermediate adversarial image saving
    dir_path = os.path.join("imagenette", "adversarial", attack_name)
    os.makedirs(dir_path, exist_ok=True)

    accuracies = []
    adv_accuracies = []
    eps_range = []

    if attack_name == "fgsm":
        eps_range = np.linspace(0, 1, eps_steps)
    else:
        eps_range = np.linspace(0, 0.3, eps_steps)

    pbar = tqdm(eps_range, desc="eps")

    for i, eps in enumerate(pbar):

        dataloader_copy = deepcopy(dataloader)

        correct = 0
        adv_correct = 0
        total = 0

        atk = get_attack(attack_name, model)
        atk = atk(model, eps=eps, **kwargs)

        for idx_batch in tqdm(range(16), desc="batches"):
            batch = next(iter(dataloader_copy))

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

        # Only visualize the last batch for comparison sake
        adv_utils.visualize_batch(dir_path, attack_name, i, images, adv_images)

        accuracy = correct / total
        adv_accuracy = adv_correct / total

        accuracies.append(accuracy)
        adv_accuracies.append(adv_accuracy)

    return accuracies, adv_accuracies, eps_range


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--attack', type=str,
                        default="fgsm",
                        help="attack name",
                        choices=["fgsm", "pgd"])
    parser.add_argument('--pgd_steps', type=int,
                        default=20,
                        help="pgd training steps")
    parser.add_argument("--eps_steps", type=int,
                        default=5, help="Eps step size")
    parser.add_argument("--latex", type=bool, default=0,
                        help="Indicate if you have latex installed on system")

    args = parser.parse_args()

    # Use cuda if cuda is avaliable
    device = torch.device("cuda") if torch.cuda.is_available else torch.device("cpu")

    # Certain styling options only availiable if latex is installed
    if args.latex:
        plt.style.use(['science'])
    else:
        plt.style.use(['science', 'no-latex'])

    # Get dataloaders
    train_loader, val_loader, train_sampler = adv_utils.get_dataloaders(16, 6)

    os.makedirs("adversarial", exist_ok=True)

    attack_name = args.attack

    # Prepare texture invariant and in distribution classifiers
    texture_inv_path = join("imagenette", "experiments", "classifier_texture_inv")
    indist_path = join("imagenette", "experiments", "classifier_indist")

    texture_inv_model, _ = adv_utils.load_model(texture_inv_path)
    indist_model, _ = adv_utils.load_model(indist_path)
    og_model = ResNet50Model(device)

    texture_inv_model = texture_inv_model.to(device)
    indist_model = indist_model.to(device)
    og_model = og_model.to(device)

    # We wrap models in one that only returns the average prediction
    texture_inv_model = AvgPredGetter(texture_inv_model)
    indist_model = AvgPredGetter(indist_model)

    inv_accuracies, inv_adv_accuracies, eps_range = (
        evaluate_model_attack(attack_name, texture_inv_model, val_loader,
                              eps_steps=args.eps_steps)
    )
    indist_accuracies, indist_adv_accuracies, eps_range = (
        evaluate_model_attack(attack_name, indist_model, val_loader,
                              eps_steps=args.eps_steps)
    )
    og_accuracies, og_adv_accuracies, eps_range = (
        evaluate_model_attack(attack_name, og_model, val_loader,
                              eps_steps=args.eps_steps)
    )

    # Plot everything and save it
    fig = plt.figure()
    plt.title(f"{attack_name.upper()} accuracy curve")
    plt.plot(eps_range, inv_adv_accuracies,
             label=f"Texture invariant (orig. acc: {np.mean(inv_accuracies):.3f})")
    plt.plot(eps_range, indist_adv_accuracies,
             label=f"In distribution (orig. acc: {np.mean(indist_accuracies):.3f})")
    plt.plot(eps_range, og_adv_accuracies,
             label=f"Resnet50 (orig. acc: {np.mean(og_accuracies):.3f})")
    plt.xlabel("Epsilon")
    plt.ylabel("Accuracy")
    plt.ylim([0, 1])
    plt.legend(loc="upper right")
    fig.set_size_inches(5, 4)

    os.makedirs('imagenette/adv_plots', exist_ok=True)
    plt.savefig(os.path.join('imagenette/adv_plots',
                f"{attack_name}_{args.eps_steps}.png"),
                dpi=100, bbox_inches="tight")
    plt.show()
