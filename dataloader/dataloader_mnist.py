import torch
import torch.nn as nn

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def training_dataset(**kwargs):
    return datasets.FashionMNIST(
                root="data",
                train=True,
                download=True,
                transform=transforms.ToTensor(),
            )

def test_dataset(**kwargs):
    return datasets.FashionMNIST(
                root="data",
                train=False,
                download=True,
                transform=transforms.ToTensor(),
            )