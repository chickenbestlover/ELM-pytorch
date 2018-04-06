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
parser.add_argument('--batch-size', type=int, default=30000, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=0, metavar='S',
                    help='random seed (default: 1)')
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
        self.conv1 = nn.Conv2d(1, 15, kernel_size=5, padding=1)
        self.conv2 = nn.Conv2d(15, 210, kernel_size=4, padding=1)

        #self.fc1 = nn.Linear(16*20, 1200, bias=True)
        self.fc2 = nn.Linear(36*210, 10, bias=False)

    def forward(self, x):

        x = self.conv1(x)
        x = F.max_pool2d(x,kernel_size=2)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x,kernel_size=2)
        x = F.relu(x)

        x = x.view(-1, self.num_flat_features(x))

        #x = self.fc1(x)
        #x = F.relu(x)

        x = self.fc2(x)

        return x

    def forwardToHidden(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x,kernel_size=2)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x,kernel_size=2)
        x = F.relu(x)
        x = x.view(-1, self.num_flat_features(x))
        #x = self.fc1(x)
        #x = F.relu(x)

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
optimizer= pseudoInverse(params=model.parameters(),C=1e-3)
#print(list(model.parameters()))




def train():
    init = time.time()
    model.train()
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)

        #print(data.size())
        hiddenOut = model.forwardToHidden(data)
        optimizer.train(inputs=hiddenOut, targets=target)
        '''
        output = model.forward(data)
        pred = output.data.max(1)[1]
        correct += pred.eq(target.data).cpu().sum()
    ending = time.time()
    print('training time: {:.2f}sec'.format(ending - init))
    print('\nTrain set accuracy: {}/{} ({:.0f}%)\n'.format(
        correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)))
        '''

def train_someBatch(batchidx=0):
    init = time.time()
    model.train()
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)

        if batch_idx == batchidx:
            hiddenOut = model.forwardToHidden(data)
            optimizer.train(inputs=hiddenOut, targets=target)
        output = model.forward(data)
        pred = output.data.max(1)[1]
        correct += pred.eq(target.data).cpu().sum()

    ending = time.time()
    print('training time: {:.2f}sec'.format(ending - init))
    print('\nTrain set accuracy: {}/{} ({:.0f}%)\n'.format(
        correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)))

def train_accuracy():
    model.train()
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        output = model.forward(data)
        pred = output.data.max(1)[1]
        correct += pred.eq(target.data).cpu().sum()
    ending = time.time()
    #print('training time: {:.2f}sec'.format(ending - init))
    print('\nTrain set accuracy: {}/{} ({:.2f}%)\n'.format(
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
        #print(pred)
        correct += pred.eq(target.data).cpu().sum()
    print('\nTest set accuracy: {}/{} ({:.2f}%)\n'.format(
        correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def train_sequential(starting_batch_idex=0):
    model.train()
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        if batch_idx >= starting_batch_idex:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data, volatile=True), Variable(target)
            hiddenOut = model.forwardToHidden(data)
            optimizer.train_sequential(hiddenOut,target)


            output=model.forward(data)
            pred=output.data.max(1)[1]
            correct += pred.eq(target.data).cpu().sum()
            print('\n{}st Batch train set accuracy: {}/{} ({:.2f}%)\n'.format(batch_idx,
                correct, (train_loader.batch_size*(batch_idx+1-starting_batch_idex)),
                100. * correct / (train_loader.batch_size*(batch_idx+1-starting_batch_idex))))

            test()

# CNN-ELM
#init=time.time()
#train()
#train_accuracy()
#test()
#ending=time.time()
#print(ending-init)

# Online Sequential CNN-ELM, batch_size is resized.
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0,), (1,))
                   ])),
    batch_size=10000, shuffle=True, **kwargs)

train_someBatch(batchidx=0) #initialize phase; offline batch training
train_sequential(starting_batch_idex=1) # Sequential learning phase; online sequential training
train_sequential(starting_batch_idex=0) # Sequential learning phase; online sequential training

