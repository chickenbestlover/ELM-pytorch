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
import numpy as np
# Training settings
parser = argparse.ArgumentParser(description='PyTorch ELM MNIST Example')
parser.add_argument('--batch-size', type=int, default=60000, metavar='N',
                    help='input batch size for training (default: 60000)')
parser.add_argument('--test-batch-size', type=int, default=10000, metavar='N',
                    help='input batch size for testing (default: 10000)')
parser.add_argument('--hidden_size', type=int, default=1000, metavar='N',
                    help='hidden size')
parser.add_argument('--activation', type=str, default='leaky_relu',
                    help='non-linear activation (default: leaky_relu')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
print('Use CUDA:',args.cuda)
print('Pytorch Version:',torch.__version__)

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
    def __init__(self,hidden_size=7000,activation='leaky_relu'):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28*28, hidden_size)
        self.activation = getattr(F,activation)
        if activation in ['relu', 'leaky_relu']:
            torch.nn.init.xavier_uniform(self.fc1.weight,gain=nn.init.calculate_gain(activation))
        else:
            torch.nn.init.xavier_uniform(self.fc1.weight, gain=1)

        self.fc2 = nn.Linear(hidden_size, 10, bias=False) # ELM do not use bias in the output layer.

    def forward(self, x):
        x = x.view(x.size(0),-1)

        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x

    def forwardToHidden(self, x):
        x = x.view(x.size(0),-1)
        x = self.fc1(x)
        x = self.activation(x)

        return x

#activations = ['leaky_relu','relu','tanhshrink','softsign','selu']
activations = ['leaky_relu']*100

models = [Net(hidden_size=args.hidden_size,activation=activation) for activation in activations]

if args.cuda:
    for model in models:
        model.cuda()

optimizers= [pseudoInverse(params=model.parameters()) for model in models]


def train(args,model,optimizer,train_loader):
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data,requires_grad=False, volatile=True), \
                       Variable(target,requires_grad=False, volatile=True)
        hiddenOut = model.forwardToHidden(data)
        optimizer.train(inputs=hiddenOut, targets=target)

def test(args,models,test_loader):
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data,requires_grad=False, volatile=True), \
                       Variable(target,requires_grad=False, volatile=True)
        preds = []
        for model in models:
            output = model.forward(data)
            pred=output.data.max(1)[1]
            preds.append(pred.unsqueeze(1))
        preds = torch.cat(preds,dim=1).cpu().numpy()
        ensembled = []
        for ans in preds:
            ensembled.append(int(np.bincount(ans).argmax()))
        ensembled = torch.LongTensor(ensembled).cuda()
        correct += ensembled.eq(target.data).cpu().sum()

    print('\nTest set accuracy: {}/{} ({:.2f}%)\n'.format(
        correct, len(test_loader.dataset),
        100. * correct / (len(test_loader.dataset))))


print('Example 1: Train SLFN using Basic ELM')
print('#hidden neuron= {}'.format(args.hidden_size))
# Basic ELM. Note that this is non-iterative learning;
# therefore batch-size is the same as # of training samples.
for i,(model,optimizer) in enumerate(zip(models,optimizers)):
    train(args,model,optimizer,train_loader)

test(args,models,test_loader)
