import os

import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Normalize, Compose
from torchvision.datasets import MNIST


def load_data_and_config():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    filepath = f"{os.getcwd()}/dataset"
    mnist = MNIST(filepath,
                  train=True,
                  download=True,
                  transform=Compose([ToTensor(), Normalize(mean=(0.5,), std=(0.5,))]))
    batch_size = 100
    data_loader = DataLoader(mnist, batch_size, shuffle=True)
    return data_loader


if __name__ == '__main__':
    load_data_and_config()