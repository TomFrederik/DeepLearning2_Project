import sys
import os
from os.path import join
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import json
from collections import OrderedDict
import torchvision

import numpy as np
import matplotlib.pyplot as plt

from typing import Union, Dict, Tuple, Any, List

import repackage
repackage.up()
from imagenette.dataloader import (get_imagenette_dls, get_cf_imagenette_dls,
                                   get_cue_conflict_dls) # , get_in9_dls
from imagenette.models import InvariantEnsemble


def plot_results(
    attack_name: str, eps_range: np.ndarray, eps_steps: int,
    accuracies: List[np.ndarray],
    adv_accuracies: List[np.ndarray],
    names: List[str]
) -> None:

    # Plot everything and save it
    fig = plt.figure()
    plt.title(f"{attack_name.upper()} accuracy curve")

    for accs, adv_accs, name in zip(accuracies, adv_accuracies, names):
        plt.plot(eps_range, adv_accs,
                 label=f"{name}: (orig. acc: {np.mean(accs):.3f})")

    plt.xlabel("Epsilon")
    plt.ylabel("Accuracy")
    plt.ylim([0, 1])
    plt.legend(loc="upper right")
    fig.set_size_inches(5, 4)

    plt.savefig(os.path.join('imagenette/adv_plots',
                             f"{attack_name}_{eps_steps}.png"),
                dpi=100, bbox_inches="tight")
    plt.show()


def visualize_batch(
    dir_path: str, attack_name: str, eps_step: int,
    images: torch.Tensor, adv_images: torch.Tensor
) -> None:

    adv_path = os.path.join(dir_path,
                            f"{eps_step}_adv.jpg")
    normal_path = os.path.join(dir_path,
                               f"{eps_step}_normal.jpg")

    torchvision.utils.save_image(adv_images.detach().cpu(),
                                 adv_path,
                                 nrow=2)
    torchvision.utils.save_image(images.detach().cpu(),
                                 normal_path,
                                 nrow=2)


def load_model(
    ckpt_path: Union[str, 'os.PathLike[str]']
) -> Tuple[nn.Module, Dict[str, Any]]:

    checkpoint = torch.load(os.path.join(ckpt_path, "checkpoint.pth"))

    with open(os.path.join(ckpt_path, "args.txt"), "r") as f:
        args = json.load(f)

    model_type = args["arch"]

    model = InvariantEnsemble(model_type, False)

    # Have to do this because of DataParallel
    new_state_dict = OrderedDict()

    state_dict = checkpoint["state_dict"]

    for k, v in state_dict.items():
        # remove `module.`
        name = k[7:]
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)

    return model, args


def get_dataloaders(batch_size: int, num_workers: int) -> Any:
    return get_imagenette_dls(False, batch_size, num_workers)


if __name__ == "__main__":
    classifier_name = "classifier_2022_05_19_07_26_texture_inv"

    path = os.path.join("imagenette", "experiments", classifier_name)

    model, args = load_model(path)

    train_loader, val_loader, train_sampler = load_dataloaders(args)
    for i, batch in enumerate(val_loader):
        inp, target = batch["ims"], batch["labels"]
        print(i)
        print("*" * 50)
        print(batch)
