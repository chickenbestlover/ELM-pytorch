import torch
from torch.autograd import Variable

class pseudoInverse(object):
    def __init__(self,params,C=1e-2,forgettingfactor=1):
        self.params=list(params)
        self.is_cuda=self.params[len(self.params)-1].is_cuda
        self.C=C
        #self.params[len(self.params)-1].data.fill_(0)
        self.w=self.params[len(self.params)-1]
        self.w.data.fill_(0)#initialize output weight as zeros
        # For sequential learning in OS-ELM
        self.dimInput=self.params[len(self.params)-1].data.size()[1]
        self.forgettingfactor=forgettingfactor
        self.M=Variable(torch.inverse(self.C*torch.eye(self.dimInput)))

        if self.is_cuda:
            self.M=self.M.cuda()

    def initialize(self):
        self.M = Variable(torch.inverse(self.C * torch.eye(self.dimInput)))

        if self.is_cuda:
            self.M = self.M.cuda()
        self.w = self.params[len(self.params) - 1]
        self.w.data.fill_(0.0)


    def train(self,inputs,targets):
        oneHotTarget=self.oneHotVectorize(targets=targets)
        numSamples=inputs.size()[0]
        dimInput=inputs.size()[1]
        dimTarget=oneHotTarget.size()[1]


        xtx= torch.mm(inputs.t(),inputs)

        I = Variable(torch.eye(dimInput))
        if self.is_cuda:
            I=I.cuda()

        self.M = Variable(torch.inverse(xtx.data+self.C*I.data))
        w = torch.mm(self.M,inputs.t())
        w = torch.mm(w,oneHotTarget)

        #self.params[len(self.params)-1].data=w.t().data
        self.w.data=w.t().data

    def train_sequential(self,inputs,targets):
        oneHotTarget = self.oneHotVectorize(targets=targets)
        numSamples = inputs.size()[0]
        dimInput = inputs.size()[1]
        dimTarget = oneHotTarget.size()[1]


        I = Variable(torch.eye(numSamples))
        if self.is_cuda:
            I = I.cuda()

        self.M = (1/self.forgettingfactor) * self.M - torch.mm((1/self.forgettingfactor) * self.M,
                                             torch.mm(inputs.t(), torch.mm(Variable(torch.inverse(I.data + torch.mm(inputs, torch.mm((1/self.forgettingfactor)* self.M, inputs.t())).data)),
                                             torch.mm(inputs, (1/self.forgettingfactor)* self.M))))


        self.w.data += torch.mm(self.M,torch.mm(inputs.t(),oneHotTarget - torch.mm(inputs,self.w.t()))).t().data


    def oneHotVectorize(self,targets):
        oneHotTarget=torch.zeros(targets.size()[0],targets.max().data[0]+1)

        for i in xrange(targets.size()[0]):
            oneHotTarget[i][targets[i].data[0]]=1

        if self.is_cuda:
            oneHotTarget=oneHotTarget.cuda()
        oneHotTarget=Variable(oneHotTarget)

        return oneHotTarget

