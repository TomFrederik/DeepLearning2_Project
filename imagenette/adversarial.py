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
import pickle

import os
from os.path import join
import argparse

import adv_utils
from typing import Any, Tuple, List, Iterable, Dict


class PredGetter(nn.Module):
    def __init__(self, model: nn.Module, head_name: str) -> None:
        super().__init__()
        self.model = model
        self.head_name = head_name
        self.lim = 3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x * self.lim * 2) - self.lim
        out = self.model(x)  # type: Dict[str, torch.Tensor]
        return out[self.head_name]


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
    pgd_steps: int = 10,
) -> Tuple[List[float], List[float], np.ndarray]:

    model.eval()
    device = torch.device("cuda") if torch.cuda.is_available else torch.device("cpu")

    # Create folder for intermediate adversarial image saving
    dir_path = os.path.join("imagenette", "adversarial", attack_name)
    os.makedirs(dir_path, exist_ok=True)

    accuracies = []
    adv_accuracies = []

    if attack_name == "fgsm":
        eps_range = np.linspace(0, 0.15, eps_steps)
    elif attack_name == "pgd":
        eps_range = np.linspace(0, 0.01, eps_steps)
    pbar = tqdm(eps_range, desc="eps")

    for i, eps in enumerate(pbar):

        dataloader_copy = deepcopy(dataloader)

        correct = 0
        adv_correct = 0
        total = 0

        atk = get_attack(attack_name, model)

        if attack_name == "pgd":
            atk = atk(model, eps=eps, steps=pgd_steps)
        else:
            atk = atk(model, eps=eps)

        for batch in tqdm(dataloader_copy, desc="batches"):
            images, labels = batch["ims"], batch["labels"]
            images, labels = images.to(device), labels.to(device)
            total += len(labels)

            trafo_images = (images + model.lim) / (2 * model.lim)

            outputs = model(trafo_images)

            trafo_adv_images = atk(trafo_images, labels)
            adv_images = (trafo_adv_images * 2 * model.lim) - model.lim

            adv_outputs = model(trafo_adv_images)

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
    parser.add_argument("--plot", type=bool, default=1,
                        help="If we should plot")

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
    head_path = join("imagenette", "experiments", "classifier_heads")
    resnet50_path = join("imagenette", "experiments", "classifier_resnet50")

    head_model, _ = adv_utils.load_model(head_path)
    resnet_model, _ = adv_utils.load_model(resnet50_path)

    head_model = head_model.to(device)
    resnet_model = resnet_model.to(device)

    # We wrap models in one that only returns the average prediction
    shape_model = PredGetter(head_model, "shape_preds")
    bg_model = PredGetter(head_model, "bg_preds")
    texture_model = PredGetter(head_model, "texture_preds")
    avg_model = PredGetter(head_model, "avg_preds")
    og_model = PredGetter(resnet_model, "avg_preds")

    shape_accs, shape_adv_accs, eps_range = (
        evaluate_model_attack(attack_name, shape_model, val_loader,
                              eps_steps=args.eps_steps, pgd_steps=args.pgd_steps)
    )
    bg_accs, bg_adv_accs, eps_range = (
        evaluate_model_attack(attack_name, bg_model, val_loader,
                              eps_steps=args.eps_steps, pgd_steps=args.pgd_steps)
    )
    texture_accs, texture_adv_accs, eps_range = (
        evaluate_model_attack(attack_name, texture_model, val_loader,
                              eps_steps=args.eps_steps, pgd_steps=args.pgd_steps)
    )
    avg_accs, avg_adv_accs, eps_range = (
        evaluate_model_attack(attack_name, avg_model, val_loader,
                              eps_steps=args.eps_steps, pgd_steps=args.pgd_steps)
    )

    og_accs, og_adv_accs, eps_range = (
        evaluate_model_attack(attack_name, og_model, val_loader,
                              eps_steps=args.eps_steps, pgd_steps=args.pgd_steps)
    )

    accuracies = [shape_accs, bg_accs, texture_accs, avg_accs, og_accs]
    adv_accuracies = [shape_adv_accs, bg_adv_accs, texture_adv_accs,
                      avg_adv_accs, og_adv_accs]
    names = ["Shape", "Background", "Texture", "Avg", "ResNet50"]

    data_dict = {"accuracies": accuracies,
                 "adv_accuracies": adv_accuracies,
                 "names": names}

    os.makedirs('imagenette/adv_plots', exist_ok=True)
    with open(join("imagenette", "adv_plots", f"{attack_name}_{args.eps_steps}.pkl"), "wb") as f:
        pickle.dump(data_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    if args.plot:
        adv_utils.plot_results(attack_name, eps_range, args.eps_steps,
                               accuracies, adv_accuracies, names)