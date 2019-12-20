import torch
from torchvision import transforms, datasets
from dataset import MNIST_ROT_PLUS

data_root = './data'

def dataloader(dataset, batch_size, test_batch_size):
    if dataset == 'MNIST':
        train_loader = torch.utils.data.DataLoader(
                datasets.MNIST(data_root, train=True, download=True,
                           transform=transforms.Compose([
                                   transforms.Resize(32),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.1307,), (0.3081,))
                               ])),
                batch_size=batch_size, shuffle=True)

        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(data_root, train=False,
                           transform=transforms.Compose([
                               transforms.Resize(32),
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=test_batch_size, shuffle=True)

        return train_loader, test_loader

    elif dataset == 'MNIST-rot':
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(data_root, train=True, download=True,
                           transform=transforms.Compose([
                               transforms.Resize(32),
                               transforms.RandomRotation(180),
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=batch_size, shuffle=True)

        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(data_root, train=False,
                           transform=transforms.Compose([
                               transforms.Resize(32),
                               transforms.RandomRotation(180),
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=test_batch_size, shuffle=True)

        return train_loader, test_loader

    elif dataset == 'MNIST-rot+':
        train_loader = torch.utils.data.DataLoader(
            MNIST_ROT_PLUS(data_root, train=True,
                           transform=transforms.Compose([
                               transforms.Resize(32),
                               transforms.RandomRotation(180),
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            MNIST_ROT_PLUS(data_root, train=False,
                           transform=transforms.Compose([
                               transforms.Resize(32),
                               transforms.RandomRotation(180),
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=test_batch_size, shuffle=True)

        return train_loader, test_loader

