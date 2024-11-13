import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import torch.nn.init as init
import optuna
import numpy as np

import torch.optim as optim

class Discriminator(nn.Module):
    def __init__(self, input_size, config:dict):
        super(Discriminator, self).__init__()
        self.input_size = input_size
        self.conf_general_encode = config["generate"]["encoder"]
        
        layer_num = self.conf_general_encode["layer_num"]
        arch = self.conf_general_encode["architecture"]
        bNorm = self.conf_general_encode["batchNorm"]
        relu = self.conf_general_encode["relu"]
        drop = self.conf_general_encode["dropout"]
        
        def res_block(in_dim, out_dim, idx):
            layers_per_block = []
            layers_per_block.append(nn.Linear(in_dim,out_dim))
            if bNorm[idx] != 0: layers_per_block.append(nn.LayerNorm(normalized_shape=bNorm[idx]))
            if relu[idx] != 0: layers_per_block.append(nn.GELU())
            if drop[idx] != 0: layers_per_block.append(nn.Dropout(drop[idx]))
            
            return layers_per_block
        
        self.body1 = nn.Sequential(*res_block(self.input_size, arch[0][0], 0))
        self.body2 = nn.Sequential(*res_block(arch[1][0],arch[1][1], 1))
        self.body3 = nn.Sequential(*res_block(arch[2][0],arch[2][1], 2))
        self.body4 = nn.Sequential(*res_block(arch[3][0],arch[3][1], 3))
        self.body5 = nn.Sequential(*res_block(arch[4][0], 1, 4))
        
        # layers = []
        # for idx in range(layer_num):
        #     if idx == 0: 
        #         layers.append(nn.Linear(self.input_size, arch[idx][0]))
        #         #init.xavier_uniform_(layers[-1].weight)
        #     elif idx == layer_num - 1: 
        #         layers.append(nn.Linear(arch[idx][0], self.zdim*2))
        #         #init.xavier_uniform_(layers[-1].weight)
        #     else: 
        #         layers.append(nn.Linear(arch[idx][0],arch[idx][1]))
        #         #init.xavier_uniform_(layers[-1].weight)
            
        #     #if bNorm[idx] != 0: layers.append(nn.BatchNorm1d(num_features=bNorm[idx]))
        #     if bNorm[idx] != 0:layers.append(nn.LayerNorm(normalized_shape=bNorm[idx]))
        #     #if bNorm[idx] != 0:layers.append(nn.InstanceNorm1d(num_features=bNorm[idx]))
        #     #if relu[idx] != 0: layers.append(nn.ReLU())
        #     if relu[idx] != 0: layers.append(nn.GELU())
        #     if drop[idx] != 0: layers.append(nn.Dropout(drop[idx]))
        
        #self.body = nn.Sequential(*layers)

        
    def forward(self, x):
        x_new = self.body1(x)
        res = x_new
        x_new = self.body2(x_new)
        #x_new = self.body3(x_new)
        #x_new = self.body4(x_new)
        #x_new += res
        scores = self.body5(x_new)
        
        probs = torch.sigmoid(scores)
        return probs

class Generator(nn.Module):
    def __init__(self, zdim, output_dim, config:dict):
        super(Generator, self).__init__()
        self.zdim = zdim
        self.output_dim = output_dim
        self.conf_general_encode = config["generate"]["encoder"]
        
        layer_num = self.conf_general_encode["layer_num"]
        arch = self.conf_general_encode["architecture"]
        bNorm = self.conf_general_encode["batchNorm"]
        relu = self.conf_general_encode["relu"]
        drop = self.conf_general_encode["dropout"]
        
        #layers = []
        
        def res_block(in_dim, out_dim, idx):
            layers_per_block = []
            layers_per_block.append(nn.Linear(in_dim,out_dim))
            if bNorm[idx] != 0: layers_per_block.append(nn.LayerNorm(normalized_shape=bNorm[idx]))
            if relu[idx] != 0: layers_per_block.append(nn.GELU())
            if drop[idx] != 0: layers_per_block.append(nn.Dropout(drop[idx]))
            
            return layers_per_block
        
        self.body1 = nn.Sequential(*res_block(self.zdim, arch[0][0], 0))
        self.body2 = nn.Sequential(*res_block(arch[1][0],arch[1][1], 1))
        self.body3 = nn.Sequential(*res_block(arch[2][0],arch[2][1], 2))
        self.body4 = nn.Sequential(*res_block(arch[3][0],arch[3][1], 3))
        self.body5 = nn.Sequential(*res_block(arch[4][0], self.output_dim, 4)) 
        
        
        
        # for idx in range(layer_num):
        #     if idx == 0: 
        #         layers.append(nn.Linear(self.zdim, arch[idx][0]))
        #         #init.xavier_uniform_(layers[-1].weight)
        #     elif idx == layer_num - 1: 
        #         layers.append(nn.Linear(arch[idx][0], self.input_size*2-1))
        #         #init.xavier_uniform_(layers[-1].weight)
        #     else: 
        #         layers.append(nn.Linear(arch[idx][0],arch[idx][1]))
        #         #init.xavier_uniform_(layers[-1].weight)
            
        #     #if bNorm[idx] != 0: layers.append(nn.BatchNorm1d(num_features=bNorm[idx]))
        #     if bNorm[idx] != 0:layers.append(nn.LayerNorm(normalized_shape=bNorm[idx]))
        #     #if bNorm[idx] != 0:layers.append(nn.InstanceNorm1d(num_features=bNorm[idx]))
        #     #if relu[idx] != 0: layers.append(nn.ReLU())
        #     if relu[idx] != 0: layers.append(nn.GELU())
        #     #if relu[idx] != 0: layers.append(nn.Sigmoid())
        #     if drop[idx] != 0: layers.append(nn.Dropout(drop[idx]))
        
        #self.body = nn.Sequential(*layers)

    def forward(self, z, feature_type_dict):
        z_new = self.body1(z)
        res = z_new
        #z_new += x_skip
        z_new = self.body2(z_new)
        #z_new = self.body3(z_new)
        #z_new = self.body4(z_new)
        #z_new += res
        xhat = self.body5(z_new)

        
        # Gaussian
        xhat_real = xhat[:, feature_type_dict['real_data']]
        # Bernoulli
        xhat_bernoulli = torch.cat([torch.sigmoid(xhat[:, val]).unsqueeze(1) for val in feature_type_dict['binary_data']], dim=1)
        # Categorical
        #xhat_categorical = torch.cat([F.softmax(xhat[:, val[0]:val[1]], dim=1) for val in feature_type_dict['categorical_param']],dim=1)
        xhat_categorical = torch.cat([F.softmax(xhat[:, val[0]:val[1]], dim=1) for val in feature_type_dict['categorical_one_hot']],dim=1)
        return torch.cat((xhat_real, xhat_bernoulli, xhat_categorical), dim=1)



class GAN(nn.Module):
    def __init__(self, prior, zdim, device, input_size, config:dict, output_dim):
        super(GAN, self).__init__()
        self.device = device

        self.zdim = zdim
        self.input_size = input_size
        self.config = config
        self.decoder_output_dim = output_dim
        
        self.discriminator = Discriminator(self.input_size, self.config).to(self.device)
        self.generator = Generator(self.zdim, self.decoder_output_dim, self.config).to(self.device)
        self.prior = prior
        self.discriminator_optimizer = optim.Adam(self.discriminator.parameters(), lr=self.config['generate']['general']["lr"])
        self.generator_optimizer = optim.Adam(self.generator.parameters(), lr=self.config['generate']['general']["lr"])
        
    def forward(self, x, feature_type_dict):
        
        x = x.to(self.device)
        x_labels = torch.ones(x.shape[0], 1).to(self.device)
        z_samples = torch.randn(x.shape[0], self.zdim).to(self.device)
        x_generated = self.generator(z_samples, feature_type_dict)
        xhat_labels = torch.zeros(x_generated.shape[0], 1).to(self.device)

        xxhat = torch.cat([x, x_generated], dim=0)
        xxhat_labels = torch.cat([x_labels, xhat_labels], dim=0)

        # Traing the discriminator
        self.discriminator.zero_grad()
        discriminator_out = self.discriminator(xxhat)
        loss_discriminator = self.loss_fn(discriminator_out, xxhat_labels)
        loss_discriminator.backward()
        self.discriminator_optimizer.step()
        
        z_samples = torch.randn(x.shape[0], self.zdim).to(self.device)
        # Training the generator
        self.generator.zero_grad()
        generator_out = self.generator(z_samples, feature_type_dict)
        discriminator_generated_out = self.discriminator(generator_out)
        loss_generator = self.loss_fn(discriminator_generated_out, x_labels)
        loss_generator.backward()
        self.generator_optimizer.step()
    
        return loss_discriminator, loss_generator

    def count_params(self):
        return sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)

    def loss_fn(self, samples, labels):
        return F.binary_cross_entropy(samples, labels)
    
    