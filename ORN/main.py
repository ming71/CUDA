import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import argparse
import os
import numpy as np
from datetime import datetime

from orn.modules import ResNet18
from orn.modules import CNN
from orn.modules import ORN
from utils import progress_bar
from dataset import dataloader

# TODO: python argument
parser = argparse.ArgumentParser("Pytorch MNIST Training")
parser.add_argument('--net', default='cnn',
                    help='Backbone for training')
parser.add_argument('--batch-size', type=int, default=128,
                    help='Input batch size for training (default: 128)')
parser.add_argument('--dataset', default='MNIST',
                    help='Dataset to test (default MNIST)')
parser.add_argument('--epochs', type=int, default=20,
                    help='Number of epochs to train (default 60)')
parser.add_argument('--test-batch-size', type=int, default=1000,
                    help='Input batch size for testing (default: 1000)')
parser.add_argument('--lr', type=int, default=1.,
                    help='Learning rate in the beginning of train (default 1)')
args = parser.parse_args()


def train(epoch):
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    total = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data.to(device)), Variable(target.to(device))
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        progress_bar(batch_idx, len(train_loader), 'Lr: %.4f | Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (optimizer.param_groups[0]['lr'], train_loss/(batch_idx+1), 100. * correct / total, correct, total))
def test(epoch):
    global best_test_acc
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = Variable(data.to(device)), Variable(target.to(device))
        output = model(data)
        loss = F.nll_loss(output, target)

        test_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        progress_bar(batch_idx, len(test_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    test_acc = 100. * correct / total
    if test_acc > best_test_acc:
        best_test_acc = test_acc
        state = {
            'acc': test_acc,
            'epoch': epoch,
            'dataset': dataset,
            'backbone': args.net,
            'net': model.state_dict()
        }
        torch.save(state, os.path.join(save_path, 'ckpt.pth'))
        print('Best accuracy %.2f' % test_acc)

if __name__ == "__main__":

    # TODO: training backbone
    print('==> Building model..')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.net == 'cnn':
        model = CNN().to(device)
        print('Backbone: CNN(baseline)')
    elif args.net == 'orn':
        model = ORN().to(device)
        print('Backbone: ORN(CNN)')
    elif args.net == 'resnet18':
        model = ResNet18().to(device)
        print('Backbone: ResNet18')

    # TODO: training and test datasets
    print('==> Preparing data..')
    dataset = args.dataset
    dataset_path = os.path.join('outputs', dataset)
    time = datetime.now().strftime('%Y-%m-%d-%H:%M')
    save_path = os.path.join(dataset_path, time)
    if not os.path.isdir(dataset_path):
        os.mkdir(dataset_path)
        os.mkdir(save_path)
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    print('Dataset:', dataset)
    print('Checkpoint save dir:', save_path)

    # TODO: datasets loader
    train_loader, test_loader = dataloader(dataset, args.batch_size, args.test_batch_size)

    # TODO: optimizer / scheduler / loss function
    optimizer = optim.Adadelta(model.parameters())
    # optimizer = optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=5e-4)
    # loss_func = nn.CrossEntropyLoss()
    # scheduler = MultiStepLR(optimizer, milestones=[180, 240], gamma=0.1)

    # TODO: train and test
    best_test_acc = 0.85

    if dataset == 'MNIST-rot+':
        epochs = int(np.ceil(args.epochs / 8))
    elif dataset == 'MNIST' or dataset == 'MNIST-rot':
        epochs = args.epochs

    print('==>Hyper-parameters')
    print('train_batch_size:', args.batch_size, ',test_batch_size:', args.test_batch_size,
          ',epochs:', epochs, ',lr:', args.lr)

    for epoch in range(1, epochs + 1):
        train(epoch)
        test(epoch)
        # scheduler.step()
    print('Best accuracy: {:.2f}%'.format(best_test_acc))
