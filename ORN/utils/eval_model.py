import torch
from torchvision import transforms, datasets
import torch.nn.functional as F
from torch.autograd import Variable
import argparse
import os

from orn.modules import ResNet18
from orn.modules import CNN
from orn.modules import ORN
from utils import progress_bar

parser = argparse.ArgumentParser("Pytorch MNIST Testing")
parser.add_argument('--dataset', default='MNIST',
                    help='Dataset to test (default MNIST)')
parser.add_argument('--model-path',
                    help='The path of the model')
parser.add_argument('--test-batch-size', type=int, default=1000,
                    help='Input batch size for testing (default: 1000)')
args = parser.parse_args()

data_root = 'data'
net_dir = args.model_path
dataset = args.dataset

if __name__ == '__main__':

    # TODO: Testing backbone
    print('==> Loading model..')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Net model dir: ', net_dir)
    paras = torch.load(net_dir)
    model_paras = paras['net']
    backbone = paras['backbone']

    if backbone == 'cnn':
        model = CNN().to(device)
        model.load_state_dict(model_paras)
        print('Backbone: CNN(baseline)')
    elif backbone == 'orn':
        model = ORN().to(device)
        model.load_state_dict(model_paras)
        print('Backbone: ORN(CNN)')



    # TODO: Test datasets
    print('==> Preparing data..')
    print('Dataset:', dataset)

    if dataset == 'MNIST':
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(data_root, train=False,
                           transform=transforms.Compose([
                               transforms.Resize(32),
                               transforms.RandomRotation(180),
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=args.test_batch_size, shuffle=True)
    else:
        raise ValueError("I have not write other dataset eval...")

    # TODO： Inference
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

    print('Model accuracy: %.2f' %test_acc)
