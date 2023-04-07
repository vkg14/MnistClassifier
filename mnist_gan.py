import os

import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Normalize, Compose
from torchvision.datasets import MNIST
from torchvision.utils import save_image

IMAGE_SIZE = 784  # 28x28 images
HIDDEN_SIZE = 256
BATCH_SIZE = 100
LATENT_SIZE = 64


def load_data_and_config():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    filepath = f"{os.getcwd()}/dataset"
    mnist = MNIST(filepath,
                  train=True,
                  download=True,
                  transform=Compose([ToTensor(), Normalize(mean=(0.5,), std=(0.5,))]))
    data_loader = DataLoader(mnist, BATCH_SIZE, shuffle=True)
    return data_loader, device


def reset_grad():
    d_optimizer.zero_grad()
    g_optimizer.zero_grad()


def train_discriminator(images):
    """
    We label our training data as real images (ones) and our generated images as fakes (zeros).
    The loss is a combination of the loss on the real images (trained) and fake images (generated).
    """

    # Loss for real images
    real_labels = torch.ones(BATCH_SIZE, 1).to(device)
    real_score = D(images)
    d_loss_real = loss_fn(real_score, real_labels)

    # Loss for fake images
    fake_labels = torch.zeros(BATCH_SIZE, 1).to(device)
    z = torch.randn(BATCH_SIZE, LATENT_SIZE).to(device)
    fake_images = G(z)
    fake_score = D(fake_images)
    d_loss_fake = loss_fn(fake_score, fake_labels)

    # Here we combine losses, reset and re-compute the gradients, and adjust model parameters using grad.
    d_loss = d_loss_real + d_loss_fake
    reset_grad()
    d_loss.backward()
    d_optimizer.step()

    return d_loss, real_score, fake_score


def train_generator():
    z = torch.randn(BATCH_SIZE, LATENT_SIZE).to(device)
    fake_images = G(z)
    labels = torch.ones(BATCH_SIZE, 1).to(device)
    # We want to minimize the loss between us believing all our images are
    # "real" quality and what the discriminator thinks.
    g_loss = loss_fn(D(fake_images), labels)

    reset_grad()
    g_loss.backward()
    g_optimizer.step()
    return g_loss, fake_images


def save_generated_image(index):
    sample_dir = 'samples'
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)
    sample_vectors = torch.randn(BATCH_SIZE, LATENT_SIZE).to(device)
    fake_images = G(sample_vectors)
    fake_images = fake_images.reshape(fake_images.size(0), 1, 28, 28)
    fake_fname = 'fake_images-{0:0=4d}.png'.format(index)
    print('Saving', fake_fname)

    def denorm(x):
        out = (x + 1) / 2
        return out.clamp(0, 1)

    save_image(denorm(fake_images), os.path.join(sample_dir, fake_fname), nrow=10)


def training_loop():
    num_epochs = 100
    total_step = len(data_loader)
    d_losses, g_losses, real_scores, fake_scores = [], [], [], []

    for epoch in range(num_epochs):
        for i, (images, _) in enumerate(data_loader):
            # Flatten images to single dimension (can we use view instead of reshape here?)
            images = images.reshape(BATCH_SIZE, -1).to(device)

            # Train the discriminator and generator
            d_loss, real_score, fake_score = train_discriminator(images)
            g_loss, fake_images = train_generator()

            # Inspect the losses
            if (i + 1) % 200 == 0:
                d_losses.append(d_loss.item())
                g_losses.append(g_loss.item())
                real_scores.append(real_score.mean().item())
                fake_scores.append(fake_score.mean().item())
                print('Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}'
                      .format(epoch, num_epochs, i + 1, total_step, d_loss.item(), g_loss.item(),
                              real_score.mean().item(), fake_score.mean().item()))

        save_generated_image(epoch + 1)


if __name__ == '__main__':
    data_loader, device = load_data_and_config()

    # This is a binary classifier which categorizes its input as "real" or "generated"
    D = nn.Sequential(
        nn.Linear(IMAGE_SIZE, HIDDEN_SIZE),
        nn.LeakyReLU(0.2),
        nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
        nn.LeakyReLU(0.2),
        nn.Linear(HIDDEN_SIZE, 1),
        nn.Sigmoid())
    D.to(device)
    # This is a classifier, given random noise of a certain latent size, generates an "image size" result.
    G = nn.Sequential(
        nn.Linear(LATENT_SIZE, HIDDEN_SIZE),
        nn.ReLU(),
        nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
        nn.ReLU(),
        nn.Linear(HIDDEN_SIZE, IMAGE_SIZE),
        nn.Tanh())
    G.to(device)
    # Binary cross entropy since Discriminator is binary classifier
    loss_fn = nn.BCELoss()
    d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0002)
    g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0002)
    training_loop()
