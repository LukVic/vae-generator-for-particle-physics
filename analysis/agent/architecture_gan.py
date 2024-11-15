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
        
    def forward(self, x):
        x_new = self.body1(x)
        res = x_new
        x_new = self.body2(x_new)
        #x_new = self.body3(x_new)
        #x_new = self.body4(x_new)
        #x_new += res
        scores = self.body5(x_new)
        
        #probs = torch.sigmoid(scores)
        probs = scores
        return probs

class Generator(nn.Module):
    def __init__(self, zdim, output_dim, config:dict):
        super(Generator, self).__init__()
        self.zdim = zdim
        self.output_dim = output_dim
        self.conf_general_encode = config["generate"]["decoder"]
        
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
        #xhat_bernoulli = torch.cat([torch.sigmoid(xhat[:, val]).unsqueeze(1) for val in feature_type_dict['binary_data']], dim=1)
        xhat_bernoulli = torch.cat([xhat[:, val].unsqueeze(1) for val in feature_type_dict['binary_data']], dim=1)
        # Categorical
        #xhat_categorical = torch.cat([F.softmax(xhat[:, val[0]:val[1]], dim=1) for val in feature_type_dict['categorical_param']],dim=1)
        xhat_categorical = torch.cat([xhat[:, val[0]:val[1]] for val in feature_type_dict['categorical_one_hot']],dim=1)
        return torch.cat((xhat_real, xhat_bernoulli, xhat_categorical), dim=1)


class GAN(nn.Module):
    def __init__(self, prior, zdim, device, input_size, config: dict, output_dim):
        super(GAN, self).__init__()
        self.device = device
        self.zdim = zdim
        self.input_size = input_size
        self.config = config
        self.decoder_output_dim = output_dim

        self.discriminator = Discriminator(self.input_size, self.config).to(self.device)
        self.generator = Generator(self.zdim, self.decoder_output_dim, self.config).to(self.device)
        self.prior = prior
        self.discriminator_optimizer = optim.Adam(self.discriminator.parameters(), lr=self.config['generate']['general']["lr"], betas=(0.0, 0.9))
        self.generator_optimizer = optim.Adam(self.generator.parameters(), lr=self.config['generate']['general']["lr"], betas=(0.0, 0.9))

        def weights_init(m):
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
        self.generator.apply(weights_init)
        self.discriminator.apply(weights_init)

    def forward(self, x, feature_type_dict):
        x = x.to(self.device)
        batch_size = x.size(0)

        # Discriminator Training
        for _ in range(5):
            self.discriminator_optimizer.zero_grad()
            z_samples = torch.randn(batch_size, self.zdim).to(self.device)
            x_generated = self.generator(z_samples, feature_type_dict)

            real_preds = self.discriminator(x)
            fake_preds = self.discriminator(x_generated.detach())

            loss_discriminator = -(torch.mean(real_preds) - torch.mean(fake_preds))
            
            gp =self.gradient_penalty(self.discriminator, x, x_generated, self.device)
            loss_discriminator += gp
            
            loss_discriminator.backward()
            self.discriminator_optimizer.step()

            # for p in self.discriminator.parameters():
            #     p.data.clamp_(-0.01, 0.01)

        # Generator Training
        self.generator_optimizer.zero_grad()
        z_samples = torch.randn(batch_size, self.zdim).to(self.device)
        x_generated = self.generator(z_samples, feature_type_dict)

        generator_preds = self.discriminator(x_generated)
        loss_generator = -torch.mean(generator_preds)
        loss_generator.backward()
        self.generator_optimizer.step()

        return loss_discriminator, loss_generator

    def count_params(self):
        generator_params = sum(p.numel() for p in self.generator.parameters() if p.requires_grad)
        discriminator_params = sum(p.numel() for p in self.discriminator.parameters() if p.requires_grad)
        return generator_params, discriminator_params
    
    
    def gradient_penalty(self, discriminator, real_data, fake_data, device, lambda_gp=10):
        batch_size = real_data.size(0)
        alpha = torch.rand(batch_size, 1, device=device)
        alpha = alpha.expand_as(real_data)

        interpolated = alpha * real_data + (1 - alpha) * fake_data
        interpolated = interpolated.to(device).requires_grad_(True)

        # Get discriminator scores for interpolated data
        interpolated_scores = discriminator(interpolated)

        # Compute gradients w.r.t. interpolated data
        gradients = torch.autograd.grad(
            outputs=interpolated_scores,
            inputs=interpolated,
            grad_outputs=torch.ones_like(interpolated_scores),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        # Compute the L2 norm of gradients
        gradients = gradients.view(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)

        # Compute the penalty
        penalty = lambda_gp * ((gradient_norm - 1) ** 2).mean()
        return penalty
    