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

def train_rnn_optimizer(optimizer_model, optimizee, data_loader, optimizer_type='rnn_prop', optimizer_opt=optim.Adam, n_epochs=10, n_rnn_batches=1, n_steps=100, n_bptt_steps=20, scaling_param=1):
    optimizer_opt = optimizer_opt(optimizer_model.parameters())
    for i in range(n_epochs):
        for j in range(n_rnn_batches):
            mnist_w = optimizee.xavier_uniform_init()

            x = torch.rand(mnist_w.size(0)+20,1)
            # x[-20:] = torch.rand(20)
            x[:-20] = mnist_w
            x = Variable(x, requires_grad=True)

            optimizer_model.init_hidden(x.size(0))

            
            c = Variable(torch.exp(2*scaling_param*torch.rand(x.size(0)-20,1)-scaling_param), requires_grad=False)
            c1 = Variable(torch.exp(2*1*torch.rand(20,1)-1), requires_grad=False)
            a = Variable(torch.exp(2*1*torch.rand(20,1)-1), requires_grad=False)

            m = Variable(torch.zeros(x.size()))
            v = Variable(torch.zeros(x.size()))
            b1 = 0.95
            b2 = 0.95
            b1t = 1
            b2t = 1
            eps = 1e-8

            for s in range(int(n_steps / n_bptt_steps)):
               # print a
                optimizer_model.h_t, optimizer_model.c_t, optimizer_model.h_t2, optimizer_model.c_t2 = Variable(optimizer_model.h_t.data), Variable(optimizer_model.c_t.data), Variable(optimizer_model.h_t2.data), Variable(optimizer_model.c_t2.data)
                x = Variable(x.data, requires_grad=True)
                # print h_t.size()
                # print h_t.volatile, c_t.volatile, h_t2.volatile, c_t2.volatile
                loss=0
                optimizer_opt.zero_grad()
                for s2 in range(n_bptt_steps):
                    data, target = next(iter((data_loader)))
                    data, target = Variable(data), Variable(target)
                    output = optimizee(data, torch.mul(c,x[:-20]))
                    nn_loss = F.nll_loss(output, target)
                    convex_loss = F.mse_loss(torch.mul(c1,x[-20:]), a, size_average=False)
                    
                    

                    if optimizer_type=='rnn_prop':
                        loss = nn_loss
                        grads = autograd.grad(loss, x, retain_graph=True)[0].view(x.size(0),-1)
                        grads.volatile = False
                        grads = grads.detach()
                        m = b1 * m + (1 - b1) * grads
                        v = b2 * v + (1 - b2) * (grads ** 2)

                        b1t *= b1
                        b2t *= b2

                        sv = torch.sqrt(v / (1 - b2t)) + eps

                        mod_grads = torch.cat((grads / sv, (m / (1 - b1t)) / sv), 1)

                        out = optimizer_model(mod_grads)

                        x = x - 0.1 * F.tanh(out)
                    
                    elif optimizer_type=='dm_optimizer':
                        loss = nn_loss+loss
                        grads = autograd.grad(loss, x, retain_graph=True)[0].view(x.size(0),-1)
                        grads.volatile = False
                        grads = grads.detach()

                        p = 10.0

                        grad_a = torch.log(torch.maximum(torch.abs(grads), torch.exp(-p))) / p
                        grad_b = torch.clip_by_value(grads * torch.exp(p), -1, 1)

                        mod_grads = torch.cat((grad_a, grad_b), 1)
                        out = optimizer_model(grads)

                        x = x + 0.1 * out
            

                loss.backward()
                optimizer_opt.step()

            if(i%1000==0):
                torch.save(optimizer_model.state_dict(), 'model_ckpt_iter_'+str(i)+optimizer_type+'.pt')

            print i,j,s,s2,nn_loss.data[0]
    torch.save(optimizer_model.state_dict(), 'model_ckpt_iter_'+str(i)+optimizer_type+'.pt')