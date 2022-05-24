import sys
import os
from os.path import join
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import json
from collections import OrderedDict

from typing import Union, Dict, Tuple, Any

import repackage
repackage.up()
from imagenette.dataloader import (get_imagenette_dls, get_cf_imagenette_dls,
                                 get_cue_conflict_dls) # , get_in9_dls
from imagenette.models import InvariantEnsemble


def visualize_preds(model: nn.Module, val_loader: DataLoader[Any]) -> None:
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


def get_dataloaders(args: Dict[str, Any]):
    return get_imagenette_dls(args["distributed"], args["batch_size"], args["workers"])


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