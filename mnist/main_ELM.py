from __future__ import print_function
import argparse
import torch
import torch.utils.data.dataloader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import time
import torch.legacy.nn as legacy_nn
from pseudoInverse import pseudoInverse
# Training settings
parser = argparse.ArgumentParser(description='PyTorch ELM MNIST Example')
parser.add_argument('--batch-size', type=int, default=60000, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                    help='input batch size for testing (default: 1000)')
#parser.add_argument('--epochs', type=int, default=10, metavar='N',
#                   help='number of epochs to train (default: 10)')
#parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
#                   help='learning rate (default: 0.01)')
#parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
#                   help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=2, metavar='S',
                    help='random seed (default: 1)')
#parser.add_argument('--log-interval', type=int, default=10, metavar='N',
#                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
print('Use CUDA:',args.cuda)


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0,), (1,))
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0,), (1,))
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28*28, 7000)
        #self.bn = nn.BatchNorm1d(6000)
        self.fc2 = nn.Linear(7000, 10, bias=False) # ELM do not use bias in the output layer.

    def forward(self, x):
        x = x.view(-1, self.num_flat_features(x))

        x = self.fc1(x)
        #x = self.bn(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

    def forwardToHidden(self, x):
        x = x.view(-1, self.num_flat_features(x))
        x = self.fc1(x)
        #x = self.bn(x)
        x = F.relu(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

model = Net()

if args.cuda:
    model.cuda()

optimizer= pseudoInverse(params=model.parameters())


def train():
    model.train()
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        hiddenOut = model.forwardToHidden(data)
        init = time.time()
        optimizer.train(inputs=hiddenOut, targets=target)
        ending = time.time()
        print('training time: {:.2f}sec'.format(ending - init))
        output = model.forward(data)
        pred=output.data.max(1)[1]
        correct += pred.eq(target.data).cpu().sum()

    print('\nTrain set accuracy: {}/{} ({:.0f}%)\n'.format(
        correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)))

def test():
    model.eval()
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model.forward(data)
        pred=output.data.max(1)[1]
        correct += pred.eq(target.data).cpu().sum()
    print('\nTest set accuracy: {}/{} ({:.0f}%)\n'.format(
        correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


train()
test()
