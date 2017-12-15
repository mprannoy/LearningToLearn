from __future__ import division
import numpy as np
import argparse

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms


torch.manual_seed(7)
dtype = torch.FloatTensor

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()

kwargs = {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)

class mnist_linear_net(nn.Module):
    def __init__(self, D_in=784, H=100, n_hidden_layers=1, n_classes=10, activation='relu'):
        super(mnist_linear_net, self).__init__()
        activations_dict = {'relu':F.relu, 'sigmoid':F.sigmoid, 'tanh':F.tanh}
        self.D_in = D_in
        self.H = H
        self.n_hl = n_hidden_layers
        self.n_c = n_classes
        self.act = activations_dict[activation]

    def forward(self, x, weights):
        x = x.view(-1, 28*28)
        count = 0

        if self.n_hl>0:
            w1 = weights[:self.D_in*self.H].view(self.D_in, self.H)
            b1 = weights[self.D_in*self.H:self.D_in*self.H+self.H].view(self.H)

            count += self.D_in*self.H+self.H
            
            w_last = weights[count:count+self.H*self.n_c].view(self.H, self.n_c)
            b_last = weights[count+self.H*self.n_c:count+self.H*self.n_c+self.n_c].view(self.n_c)

            count += self.H*self.n_c+self.n_c

            w_mid = [None]*(self.n_hl-1)
            b_mid = [None]*(self.n_hl-1)

            for l in range(self.n_hl-1):
                w_mid[l] = weights[count:count+self.H*self.H].view(self.H, self.H)
                b_mid[l] = weights[count+self.H*self.H:count+self.H*self.H+self.H].view(self.H)
                count += self.H*self.H+self.H

        else:
            w1 = weights[:self.D_in*self.n_c].view(self.D_in, self.n_c)
            b1 = weights[self.D_in*self.n_c:self.D_in*self.n_c+self.n_c].view(self.n_c)


        x = F.linear(x, w1.t(), b1)

        if self.n_hl>0:
            x = self.act(x)
            for l in range(self.n_hl-1):
                x = self.act(F.linear(x, w_mid[l].t(), b_mid[l]))
            x = F.linear(x, w_last.t(), b_last)
        return F.log_softmax(x)

    def xavier_uniform_init(self):

        if self.n_hl == 0:
            x_dim = self.D_in * self.n_c + self.n_c
        else:
            x_dim = self.D_in * self.H + self.H + (self.H * self.H + self.H) * (self.n_hl - 1) + self.H * self.n_c + self.n_c

        x = torch.FloatTensor(x_dim,1)

        if self.n_hl>0:
            count = 0

            # nn.init.xavier_uniform(torch.FloatTensor(D_in, H)).view(-1)
            var = np.sqrt(2*2/(self.D_in+self.H))
            x[:self.D_in*self.H] = nn.init.xavier_uniform(torch.FloatTensor(self.D_in, self.H)).view(-1)
            x[self.D_in*self.H:self.D_in*self.H+self.H] = torch.FloatTensor(self.H).uniform_(-var, var)

            count += self.D_in*self.H+self.H

            var = np.sqrt(2*2/(10+self.H))
            x[count:count+self.H*self.n_c] = nn.init.xavier_uniform(torch.FloatTensor(self.H,self.n_c)).view(-1)
            x[count+self.H*self.n_c:count+self.H*self.n_c+self.n_c] = torch.FloatTensor(self.n_c).uniform_(-var, var)

            count += self.H*self.n_c+self.n_c

            for l in range(self.n_hl-1):
                var = np.sqrt(2*2/(self.H + self.H))
                x[count:count+self.H*self.H] = nn.init.xavier_uniform(torch.FloatTensor(self.H,self.H)).view(-1)
                x[count+self.H*self.H:count+self.H*self.H+self.H] = torch.FloatTensor(self.H).uniform_(-var, var)
                count += self.H*self.H+self.H

        else:
            var = np.sqrt(2*2/(self.D_in+self.n_c))
            x[:self.D_in*self.n_c] = nn.init.xavier_uniform(torch.FloatTensor(self.D_in, self.n_c)).view(-1)
            x[self.D_in*self.n_c : self.D_in*self.n_c+self.n_c] = torch.FloatTensor(self.n_c).uniform_(-var, var)

        return x

class rnn_optimizer(nn.Module):
    """docstring for rnn_optimizer"""
    def __init__(self, n_in=1, n_h=20, n_out=1):
        super(rnn_optimizer, self).__init__()

        self.n_h = n_h
        self.lstm1 = nn.LSTMCell(n_in, self.n_h)
        self.lstm2 = nn.LSTMCell(self.n_h, self.n_h)
        self.linear = nn.Linear(self.n_h, n_out)

    def forward(self, input_t, h_t, c_t, h_t2, c_t2):

        # print h_t.volatile, c_t.volatile, h_t2.volatile, c_t2.volatile
        h_t, c_t = self.lstm1(input_t, (h_t, c_t))
        h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
        output = self.linear(h_t2)

        return output, h_t, c_t, h_t2, c_t2

    def init_hidden(self, batch_size):
        h_t = Variable(torch.zeros(batch_size, self.n_h))
        c_t = Variable(torch.zeros(batch_size, self.n_h))
        h_t2 = Variable(torch.zeros(batch_size, self.n_h))
        c_t2 = Variable(torch.zeros(batch_size, self.n_h))

        return h_t, c_t, h_t2, c_t2



rnn = rnn_optimizer(n_in=2)

optimizer = optim.Adam(rnn.parameters())
optimizee = mnist_linear_net(H=20, activation='sigmoid')


n_epochs = 10
n_rnn_batches = 1
n_steps = 100
n_bptt_steps = 20

for i in range(n_epochs):
    for j in range(n_rnn_batches):
        mnist_w = optimizee.xavier_uniform_init()

        x = torch.rand(mnist_w.size(0)+20,1)
        # x[-20:] = torch.rand(20)
        x[:-20] = mnist_w
        x = Variable(x, requires_grad=True)

        h_t, c_t, h_t2, c_t2 = rnn.init_hidden(x.size(0))

        
        c = Variable(torch.exp(2*3*torch.rand(x.size(0)-20,1)-3), requires_grad=False)
        c1 = Variable(torch.exp(2*1*torch.rand(20,1)-1), requires_grad=False)
        a = Variable(torch.exp(2*1*torch.rand(20,1)-1), requires_grad=False)

        m = Variable(torch.zeros(x.size()))
        v = Variable(torch.zeros(x.size()))
        b1 = 0.95
        b2 = 0.95
        b1t = 1
        b2t = 1
        eps = 1e-8

        # print x.size()



        for s in range(int(n_steps / n_bptt_steps)):
            print a
            h_t, c_t, h_t2, c_t2, x = Variable(h_t.data, requires_grad=True), Variable(c_t.data, requires_grad=True), Variable(h_t2.data, requires_grad=True), Variable(c_t2.data, requires_grad=True), Variable(x.data, requires_grad=True)
            # print h_t.size()
            # print h_t.volatile, c_t.volatile, h_t2.volatile, c_t2.volatile

            optimizer.zero_grad()
            for s2 in range(n_bptt_steps):
                data, target = next(iter((train_loader)))
                data, target = Variable(data), Variable(target)
                output = optimizee(data, torch.mul(c,x[:-20]))
                loss = F.mse_loss(torch.mul(c1,x[-20:]), a, size_average=False) + F.nll_loss(output, target)
                grads = autograd.grad(loss, x, retain_graph=True)[0].view(x.size(0),-1)
                grads.volatile = False
                # print grads.volatile, data.volatile
                grads = grads.detach()
                print i,j,s,s2, loss#,grads

                m = b1 * m + (1 - b1) * grads
                v = b2 * v + (1 - b2) * (grads ** 2)

                b1t *= b1
                b2t *= b2

                sv = torch.sqrt(v / (1 - b2t)) + eps

                mod_grads = torch.cat((grads / sv, (m / (1 - b1t)) / sv), 1)

                out, h_t, c_t, h_t2, c_t2 = rnn(mod_grads, h_t, c_t, h_t2, c_t2)
                # out, h_t, c_t, h_t2, c_t2 = rnn(grads, h_t, c_t, h_t2, c_t2)

                # print x.volatile, out.volatile
                x = x - 0.1*F.tanh(out)
                # x -= out
                # print x.volatile
            loss.backward()
            optimizer.step()