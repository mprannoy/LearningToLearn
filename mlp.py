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

class mlp(nn.Module):
    def __init__(self, D_in=784, H=100, n_hidden_layers=1, n_classes=10, activation='relu'):
        super(mlp, self).__init__()
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