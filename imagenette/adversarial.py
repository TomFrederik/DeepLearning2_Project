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

import os
from os.path import join
import argparse

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
    dataloader: DataLoader[Any], 
    classifier_name,
    **kwargs: Any
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

        for idx_batch in tqdm(range(16), desc="batches"):
            batch = next(iter(dataloader))

            images, labels = batch["ims"], batch["labels"]
            images, labels = images.to(device), labels.to(device)
            total += len(labels)

            outputs = model(images)

            adv_images = atk(images, labels)
            
            if eps == eps_range[-1]:
                dir_path = os.path.join("imagenette", "adversarial", classifier_name, attack_name)
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path)

                adv_path = os.path.join(dir_path,
                                        f"{idx_batch}_adv.jpg")
                normal_path = os.path.join(dir_path, 
                                        f"{idx_batch}_normal.jpg")
                

                torchvision.utils.save_image(adv_images.detach().cpu(),
                                             adv_path,
                                             nrow=2)
                torchvision.utils.save_image(images.detach().cpu(),
                                             normal_path,
                                             nrow=2)
            # for i in range(adv_images.shape[0]):
            #     img = adv_images[i]
                
            #     adv_path = os.path.join("imagenette", "adversarial", 
            #                         f"{idx_batch}_{labels}")
            #     torchvision.utils.save_image(img.detach().cpu(),
            #                                 adv_path,
            #                                 normalize=True)

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

    parser = argparse.ArgumentParser()

    parser.add_argument('--classifier',type = str,
                        default= "classifier_2022_05_25_16_39_indist", 
                        help = "classifier name")

    parser.add_argument('--pgd_steps',type = int,
                        default= 20, 
                        help = "pgd training steps")

    paras = parser.parse_args()

    os.makedirs("adversarial", exist_ok=True)

    classifier_name = paras.classifier

    path = join("imagenette", "experiments", classifier_name)

    model, args = adv_utils.load_model(path)

    device = torch.device("cuda") if torch.cuda.is_available else torch.device("cpu")

    model = model.to(device)

    model = AvgPredGetter(model)

    args["batch_size"] = 16
    train_loader, val_loader, train_sampler = adv_utils.get_dataloaders(args)

    pgd_accuracies, pgd_adv_accuracies, eps_range = evaluate_model_attack("pgd", model, val_loader, classifier_name, steps=paras.pgd_steps)
    fgsm_accuracies, fgsm_adv_accuracies, eps_range = evaluate_model_attack("fgsm", model, val_loader, classifier_name)

    wandb.init(project="DL2", entity="xinyichen", config = args, 
               name ="adversarial" + "_" + str(eps_range))

    fig = plt.figure()
    plt.title(f"PGD and FGSM accuracy")
    plt.plot(eps_range, pgd_adv_accuracies, label=f"PGD Adv. Acc. (orig. acc: {np.mean(pgd_accuracies):.3f})")
    plt.plot(eps_range, fgsm_adv_accuracies, label=f"FGSM Adv. Acc. (orig. acc: {np.mean(fgsm_accuracies):.3f})")
    plt.xlabel("eps")
    plt.ylabel("accuracy")
    plt.ylim([0, 1])
    plt.legend()
    plt.show()

    if not os.path.exists('adv_plots'):
        os.makedirs('adv_plots')
    plt.savefig(os.path.join('adv_plots',\
        f"{classifier_name.split('_')[-1]}_{str(paras.pgd_steps)}_{str(len(eps_range))}.jpg"))

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
