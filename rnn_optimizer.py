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

class rnn_optimizer(nn.Module):
    """docstring for rnn_optimizer"""
    def __init__(self, n_in=1, n_h=20, n_out=1, preprocessing=False):
        super(rnn_optimizer, self).__init__()

        self.n_h = n_h
        self.preprocessing = preprocessing
        if self.preprocessing:
            self.linear1 = nn.Linear(n_in, self.n_h) #Preprocessing layer
            n_in = 20
        self.lstm1 = nn.LSTMCell(n_in, self.n_h)
        self.lstm2 = nn.LSTMCell(self.n_h, self.n_h)
        self.linear = nn.Linear(self.n_h, n_out)

    def forward(self, input_t):

        # print h_t.volatile, c_t.volatile, h_t2.volatile, c_t2.volatile
        if self.preprocessing:
            self.h_t, self.c_t = self.lstm1(F.elu(self.linear1(input_t)), (self.h_t, self.c_t))
        else:
            self.h_t, self.c_t = self.lstm1(input_t, (self.h_t, self.c_t))
        self.h_t2, self.c_t2 = self.lstm2(self.h_t, (self.h_t2, self.c_t2))
        output = self.linear(self.h_t2)

        return output

    def init_hidden(self, batch_size):
        self.h_t = Variable(torch.zeros(batch_size, self.n_h))
        self.c_t = Variable(torch.zeros(batch_size, self.n_h))
        self.h_t2 = Variable(torch.zeros(batch_size, self.n_h))
        self.c_t2 = Variable(torch.zeros(batch_size, self.n_h))