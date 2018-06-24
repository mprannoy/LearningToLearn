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

from mlp import mlp
from rnn_optimizer import rnn_optimizer
from train_optimizer import train_rnn_optimizer

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

rnn = rnn_optimizer(n_in=2,preprocessing=True)
optimizee = mlp(H=20, activation='sigmoid')

n_epochs = 50
n_rnn_batches = 1
n_steps = 100
n_bptt_steps = 20

train_rnn_optimizer(optimizer_model=rnn, optimizee=optimizee, 
                    data_loader=train_loader, optimizer_type='rnn_prop', 
                    optimizer_opt=optim.Adam , n_epochs=n_epochs, n_rnn_batches=1,
                    n_steps=100, n_bptt_steps=20, scaling_param=3)
