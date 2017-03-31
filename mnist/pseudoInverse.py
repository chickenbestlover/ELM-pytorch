import torch
from torch.autograd import Variable

class pseudoInverse(object):
    def __init__(self,params):
        self.params=list(params)
        self.is_cuda=self.params[len(self.params)-1].is_cuda

    def train(self,inputs,targets):
        oneHotTarget=self.oneHotVectorize(targets=targets)
        numSamples=inputs.size()[0]
        dimInput=inputs.size()[1]
        dimTarget=oneHotTarget.size()[1]


        xtx= torch.mm(inputs.t(),inputs)

        I = Variable(torch.eye(dimInput))
        if self.is_cuda:
            I=I.cuda()
        w = Variable(torch.inverse(xtx.data+0.0001*I.data))
        w = torch.mm(w,inputs.t())
        w = torch.mm(w,oneHotTarget)
        self.params[len(self.params)-1].data=w.t().data




    def oneHotVectorize(self,targets):
        oneHotTarget=torch.zeros(targets.size()[0],targets.max().data[0]+1)

        for i in xrange(targets.size()[0]):
            oneHotTarget[i][targets[i].data[0]]=1

        if self.is_cuda:
            oneHotTarget=oneHotTarget.cuda()
        oneHotTarget=Variable(oneHotTarget)

        return oneHotTarget

