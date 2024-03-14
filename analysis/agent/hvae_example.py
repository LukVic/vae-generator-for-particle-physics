import os

import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.datasets import load_digits
from sklearn import datasets
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F

class HierarchicalVAE(nn.Module):
    def __init__(self, nn_r_1, nn_r_2, nn_delta_1, nn_delta_2, nn_z_1, nn_x, num_vals=256, D=64, L=16, likelihood_type='categorical'):
        super(HierarchicalVAE, self).__init__()

        print('Hierachical VAE by JT.')
        
        # bottom-up path
        self.nn_r_1 = nn_r_1
        self.nn_r_2 = nn_r_2
        
        self.nn_delta_1 = nn_delta_1
        self.nn_delta_2 = nn_delta_2
        
        # top-down path
        self.nn_z_1 = nn_z_1
        self.nn_x = nn_x

        
        # other params
        self.D = D
        
        self.L = L
        
        self.num_vals = num_vals
        
        self.likelihood_type = likelihood_type
        
    def reparameterization(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mu + std * eps
        
    def forward(self, x, reduction='avg'):
        #=====
        # bottom-up
        # step 1
        r_1 = self.nn_r_1(x)
        r_2 = self.nn_r_2(r_1)
        
        #step 2
        delta_1 = self.nn_delta_1(r_1)
        delta_mu_1, delta_log_var_1 = torch.chunk(delta_1, 2, dim=1)
        delta_log_var_1 = F.hardtanh(delta_log_var_1, -7., 2.)
        
        # step 3
        delta_2 = self.nn_delta_2(r_2)
        delta_mu_2, delta_log_var_2 = torch.chunk(delta_2, 2, dim=1)
        delta_log_var_2 = F.hardtanh(delta_log_var_2, -7., 2.)
        
        # top-down
        # step 4
        z_2 = self.reparameterization(delta_mu_2, delta_log_var_2)
        
        # step 5
        h_1 = self.nn_z_1(z_2)
        mu_1, log_var_1 = torch.chunk(h_1, 2, dim=1)
        
        # step 6
        z_1 = self.reparameterization(mu_1 + delta_mu_1, log_var_1 + delta_log_var_1)
        
        # step 7
        h_d = self.nn_x(z_1)

        if self.likelihood_type == 'categorical':
            b = h_d.shape[0]
            d = h_d.shape[1]//self.num_vals
            h_d = h_d.view(b, d, self.num_vals)
            mu_d = torch.softmax(h_d, 2)

        elif self.likelihood_type == 'bernoulli':
            mu_d = torch.sigmoid(h_d)
        
        #=====ELBO
        # RE
        if self.likelihood_type == 'categorical':
            RE = log_categorical(x, mu_d, num_classes=self.num_vals, reduction='sum', dim=-1).sum(-1)

        elif self.likelihood_type == 'bernoulli':
            RE = log_bernoulli(x, mu_d, reduction='sum', dim=-1)
        
        # KL
        KL_z_2 = 0.5 * (delta_mu_2**2 + torch.exp(delta_log_var_2) - delta_log_var_2 - 1).sum(-1)
        KL_z_1 = 0.5 * (delta_mu_1**2 / torch.exp(log_var_1) + torch.exp(delta_log_var_1) -\
                        delta_log_var_1 - 1).sum(-1)
        
        KL = KL_z_1 + KL_z_2
        
        error = 0
        if np.isnan(RE.detach().numpy()).any():
            print('RE {}'.format(RE))
            print('KL {}'.format(KL))
            error = 1
        if np.isnan(KL.detach().numpy()).any():
            print('RE {}'.format(RE))            
            print('KL {}'.format(KL))
            error = 1

        if error == 1:
            raise ValueError()
        
        # Final ELBO
        if reduction == 'sum':
            loss = -(RE - KL).sum()
        else:
            loss = -(RE - KL).mean()
        
        return loss