import os
from time import time

from torch import nn, utils, optim
import torch
from torchvision import transforms, datasets

import matplotlib.pyplot as plt


def prep_dataset():
    batch_size_train = 64
    batch_size_test = 1000

    random_seed = 1
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)

    # mean = 0.1307
    # std = 0.3081
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,)),
                                    ])

    filepath = f"{os.getcwd()}/dataset"
    trainset = datasets.MNIST(filepath, download=True, train=True, transform=transform)

    testset = datasets.MNIST(filepath, download=True, train=False, transform=transform)

    train_loader = utils.data.DataLoader(trainset, batch_size=batch_size_train, shuffle=True)
    test_loader = utils.data.DataLoader(testset, batch_size=batch_size_test, shuffle=True)

    return train_loader, test_loader


class MnistModel(nn.Module):
    def __init__(self):
        super(MnistModel, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.max_pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.1)
        self.dense1 = nn.Linear(320, 120)
        self.dense2 = nn.Linear(120, 84)
        self.output_layer = nn.Linear(84, 10)

    def forward(self, x):
        # ReLu(MaxPool(Conv)) = MaxPool(ReLu(Conv)) -> they are commutative
        x = self.relu(self.max_pool(self.conv1(x)))
        x = self.relu(self.max_pool(self.conv2(x)))
        x = self.dropout(x)
        # Flatten the view
        x = x.view(-1, 320)
        x = self.relu(self.dense1(x))
        x = self.relu(self.dense2(x))
        # We normally would soft max the output layer but its captured by our loss function
        return self.output_layer(x)


def training_loop(model, loss_fn, trainloader, testloader):
    learning_rate = 0.01
    momentum = 0.9
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    start = time()
    epochs = 10
    print(f"Length of training set: {len(trainloader.dataset)}.")
    for e in range(epochs):
        running_loss = 0
        for images, labels in trainloader:
            optimizer.zero_grad()

            output = model(images)
            loss = loss_fn(output, labels)

            loss.backward()  # Back-prop

            optimizer.step()
            running_loss += loss.item()
        else:
            print(f"After epoch {e+1} -> Training loss: {running_loss}")
            test_model(testloader, model)
    print(f"\nTook {(time() - start) / 60} minutes to train with {epochs} epochs.")


def test_model(testloader, model):
    model.train()
    correct_count, all_count = 0, 0
    for images, labels in testloader:
        for i in range(len(labels)):
            # img = images[i].view(1, 784)
            with torch.no_grad():
                logps = model(images[i])

            ps = torch.exp(logps)
            probab = list(ps.numpy()[0])
            pred_label = probab.index(max(probab))
            true_label = labels.numpy()[i]
            if true_label == pred_label:
                correct_count += 1
            all_count += 1

    print(f"Number Of Images Tested = {all_count}")
    print(f"Model Accuracy = {correct_count / all_count}\n")


def see_example_images(train_loader):
    dataiter = iter(train_loader)
    images, labels = next(dataiter)

    print(images.shape)
    print(labels.shape)
    for index in range(15):
        plt.subplot(3, 5, index + 1)
        plt.axis('off')
        plt.imshow(images[index].numpy().squeeze(), cmap='gray_r')
    plt.show()


if __name__ == '__main__':
    train_loader, test_loader = prep_dataset()
    model = MnistModel()
    # Soft max + negative log loss
    loss_fn = nn.CrossEntropyLoss()
    # loss_fn = nn.NLLLoss()
    test_model(test_loader, model)
    training_loop(model, loss_fn, train_loader, test_loader)