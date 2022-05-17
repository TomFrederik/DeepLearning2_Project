import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import torchvision
from torchvision import datasets, models, transforms

import torchattacks
from torchattacks import PGD, FGSM

import matplotlib.pyplot as plt

import utils


batch_size = 32

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
loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True)

atk = torchattacks.PGD(model, eps=8/255, alpha=2/255, steps=4)

correct = 0
adv_correct = 0
total = 0

fig = plt.figure()

for inputs, labels in loader:

    total += batch_size

    outputs = model(inputs)

    adv_images = atk(inputs, labels)
    adv_outputs = model(adv_images)

    _, preds = torch.max(outputs, 1)
    _, adv_preds = torch.max(adv_outputs, 1)

    correct += (preds == labels).sum()
    adv_correct += (adv_preds == labels).sum()

    images_so_far = 0
    fig = plt.gcf()

    for j in range(inputs.shape[0]):
        images_so_far += 1
        ax = plt.subplot(4, 8, images_so_far)
        ax.axis("off")
        ax.set_title(f"{preds[j]}")
        utils.imshow(inputs.cpu().data[j])

    break

print('Accuracy:      %.2f %%' % (100 * float(correct) / total))
print('Adv. accuracy: %.2f %%' % (100 * float(adv_correct) / total))
plt.show()
