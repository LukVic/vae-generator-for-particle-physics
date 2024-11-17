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
        
        probs = torch.sigmoid(scores)
        #probs = scores
        return probs

class Critic(nn.Module):
    def __init__(self, input_size, config:dict):
        super(Critic, self).__init__()
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

        # Define the discriminator and generator
        self.discriminator = Discriminator(self.input_size, self.config).to(self.device)  # Critic renamed as Discriminator
        self.generator = Generator(self.zdim, self.decoder_output_dim, self.config).to(self.device)
        self.prior = prior

        # Use Adam optimizer with typical GAN settings
        self.discriminator_optimizer = optim.Adam(
            self.discriminator.parameters(), lr=self.config['generate']['general']["lr"], betas=(0.5, 0.999)
        )
        self.generator_optimizer = optim.Adam(
            self.generator.parameters(), lr=self.config['generate']['general']["lr"], betas=(0.5, 0.999)
        )

        # Initialize weights
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
        self.discriminator_optimizer.zero_grad()
        z_samples = torch.randn(batch_size, self.zdim).to(self.device)
        x_generated = self.generator(z_samples, feature_type_dict)

        # Real and fake predictions
        real_preds = self.discriminator(x)
        fake_preds = self.discriminator(x_generated.detach())

        # Compute BCE loss for the discriminator
        real_labels = torch.ones(batch_size, 1).to(self.device)
        fake_labels = torch.zeros(batch_size, 1).to(self.device)
        loss_real = nn.BCEWithLogitsLoss()(real_preds, real_labels)
        loss_fake = nn.BCEWithLogitsLoss()(fake_preds, fake_labels)
        loss_discriminator = loss_real + loss_fake

        loss_discriminator.backward()
        self.discriminator_optimizer.step()

        # Generator Training
        self.generator_optimizer.zero_grad()
        z_samples = torch.randn(batch_size, self.zdim).to(self.device)
        x_generated = self.generator(z_samples, feature_type_dict)

        # Generator wants the discriminator to think generated data is real
        generator_preds = self.discriminator(x_generated)
        loss_generator = nn.BCEWithLogitsLoss()(generator_preds, real_labels)

        loss_generator.backward()
        self.generator_optimizer.step()

        return loss_discriminator, loss_generator

class WGAN_GP(nn.Module):
    def __init__(self, prior, zdim, device, input_size, config: dict, output_dim):
        super(WGAN_GP, self).__init__()
        self.device = device
        self.zdim = zdim
        self.input_size = input_size
        self.config = config
        self.decoder_output_dim = output_dim

        self.critic = Critic(self.input_size, self.config).to(self.device)
        self.generator = Generator(self.zdim, self.decoder_output_dim, self.config).to(self.device)
        self.prior = prior
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.config['generate']['general']["lr"], betas=(0.0, 0.9))
        self.generator_optimizer = optim.Adam(self.generator.parameters(), lr=self.config['generate']['general']["lr"], betas=(0.0, 0.9))

        def weights_init(m):
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
        self.generator.apply(weights_init)
        self.critic.apply(weights_init)

    def forward(self, x, feature_type_dict):
        x = x.to(self.device)
        batch_size = x.size(0)

        # Critic Training
        for _ in range(5):
            self.critic_optimizer.zero_grad()
            z_samples = torch.randn(batch_size, self.zdim).to(self.device)
            x_generated = self.generator(z_samples, feature_type_dict)

            real_preds = self.critic(x)
            fake_preds = self.critic(x_generated.detach())

            loss_critic = -(torch.mean(real_preds) - torch.mean(fake_preds))
            
            gp =self.gradient_penalty(self.critic, x, x_generated, self.device)
            loss_critic += gp
            
            loss_critic.backward()
            self.critic_optimizer.step()

            # for p in self.critic.parameters():
            #     p.data.clamp_(-0.01, 0.01)

        # Generator Training
        self.generator_optimizer.zero_grad()
        z_samples = torch.randn(batch_size, self.zdim).to(self.device)
        x_generated = self.generator(z_samples, feature_type_dict)

        generator_preds = self.critic(x_generated)
        loss_generator = -torch.mean(generator_preds)
        loss_generator.backward()
        self.generator_optimizer.step()

        return loss_critic, loss_generator

    def count_params(self):
        generator_params = sum(p.numel() for p in self.generator.parameters() if p.requires_grad)
        critic_params = sum(p.numel() for p in self.critic.parameters() if p.requires_grad)
        return generator_params, critic_params
    
    
    def gradient_penalty(self, critic, real_data, fake_data, device, lambda_gp=10):
        batch_size = real_data.size(0)
        alpha = torch.rand(batch_size, 1, 1, 1, device=device) if real_data.dim() > 2 else torch.rand(batch_size, 1, device=device)
        interpolated = (alpha * real_data + (1 - alpha) * fake_data).requires_grad_(True)

        # Get critic scores for interpolated data
        interpolated_scores = critic(interpolated)

        # Compute gradients w.r.t. interpolated data
        gradients = torch.autograd.grad(
            outputs=interpolated_scores,
            inputs=interpolated,
            grad_outputs=torch.ones_like(interpolated_scores),
            create_graph=True,
            only_inputs=True
        )[0]

        # Compute the L2 norm of gradients
        gradients = gradients.view(batch_size, -1)
        gradient_norm = torch.linalg.norm(gradients, ord=2, dim=1)

        # Compute the penalty
        penalty = lambda_gp * ((gradient_norm - 1) ** 2).mean()
        return penalty
    