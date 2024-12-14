import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import torch.nn.init as init
import numpy as np
import matplotlib.pyplot as plt
import math

class Encoder(nn.Module):
    def __init__(self, zdim, input_size, config:dict):
        super(Encoder, self).__init__()
        self.zdim = zdim
        self.input_size = input_size
        self.conf_general_encode = config["generate"]["backward_diffusion_encoder"]
        
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
        self.body5 = nn.Sequential(*res_block(arch[4][0], self.zdim*2, 4))

    def sample(self, mu, std):
        qz_gauss = torch.distributions.Normal(mu, std)
        z = qz_gauss.rsample()
        return z, qz_gauss
    
    def log_probas(self,z, mu, std):
        qzx = Normal(mu,  std)
        E_log_pxz = qzx.log_prob(z).sum(dim=1)
        return E_log_pxz, qzx
        
    def forward(self, x):
        x_new = self.body1(x)
        res = x_new
        x_new = self.body2(x_new)
        #x_new = self.body3(x_new)
        #x_new = self.body4(x_new)
        #x_new += res
        scores = self.body5(x_new)
        
        #scores = self.body(x)
        mu, sigma = torch.split(scores, self.zdim, dim=1)
        std = torch.exp(sigma) 
        return mu, std

class Decoder(nn.Module):
    def __init__(self, zdim, input_size, config:dict, output_dim):
        super(Decoder, self).__init__()
        self.zdim = zdim
        self.output_dim = output_dim
        self.input_size = input_size
        self.conf_general_decode = config["generate"]["backward_diffusion_decoder"]
        
        layer_num = self.conf_general_decode["layer_num"]
        arch = self.conf_general_decode["architecture"]
        bNorm = self.conf_general_decode["batchNorm"]
        relu = self.conf_general_decode["relu"]
        drop = self.conf_general_decode["dropout"]
        
        #layers = []
        
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
        self.body5 = nn.Sequential(*res_block(arch[4][0], self.output_dim, 4)) 
        
    
    def log_probas(self,x, mu, std):
        pxz = Normal(mu,  std)
        E_log_pxz = pxz.log_prob(x).sum(dim=1)
        return E_log_pxz, pxz

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
        xhat_gauss = torch.cat([xhat[:, val[0]:val[1]] for val in feature_type_dict['real_param']], dim=1)
        xhat_gauss_mu, xhat_gauss_sigma = torch.split(xhat_gauss, len(feature_type_dict['real_data']), dim=1)
        xhat_gauss_std = torch.exp(xhat_gauss_sigma)
        # Bernoulli
        xhat_bernoulli = torch.cat([torch.sigmoid(xhat[:, val[0]]).unsqueeze(1) for val in feature_type_dict['binary_param']], dim=1)
        # Categorical
        #xhat_categorical = torch.cat([F.softmax(xhat[:, val[0]:val[1]], dim=1) for val in feature_type_dict['categorical_param']],dim=1)
        xhat_categorical = torch.cat([xhat[:, val[0]:val[1]] for val in feature_type_dict['categorical_param']],dim=1)
        return xhat_gauss_mu, xhat_gauss_std, xhat_bernoulli, xhat_categorical



class DDGM(nn.Module):
    def __init__(self, prior, zdim, device, input_size, config:dict, output_dim):
        super(DDGM, self).__init__()
        self.device = device

        self.zdim = zdim
        self.input_size = input_size
        self.config = config
        self.decoder_output_dim = output_dim
        
        self.hidden_dim = 512
        self.timesteps = 10
        #self.beta = self.linear_beta_scheduler(self.timesteps)
        self.beta = torch.FloatTensor([0.9]).to(self.device)

        self.encoders = [Encoder(self.zdim, self.input_size, self.config).to(self.device) for _ in range(self.timesteps-1)]
        self.decoder = Decoder(self.zdim, self.input_size, self.config, self.decoder_output_dim).to(self.device)
        self.prior = prior

        # Initialize weights
        def weights_init(m):
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
        self.decoder.apply(weights_init)
        for i in range(len(self.encoders)):
            self.encoders[i].apply(weights_init)

    def forward(self, x, feature_type_dict):
        #! TIME EMBEDDING
        x = x.to(self.device)
        zs  = [self.reparameterization_gaussian_diffusion(x,0)]   
        #! ENCODER
        for i in range(1, self.timesteps):
            zs.append(self.reparameterization_gaussian_diffusion(zs[-1], i))

        for i in range(1, self.timesteps):
            zs.append(self.reparameterization_gaussian_diffusion(zs[-1], i))

        mus = []
        stds = []
        for i in range(len(self.encoders) -1, -1, -1):
            mu_i, std_i = self.encoders[i](zs[i+1].view(-1, self.input_size))
            mus.append(mu_i)
            stds.append(std_i)

        #pz_gauss = torch.distributions.Normal(torch.zeros_like(mus[0]), torch.ones_like(stds[0]))
        #! DECODER
        mu_gauss, std_gauss, p_bernoulli, p_categorical = self.decoder(zs[0], feature_type_dict)
        #print(torch.sum(p_categorical[:,0:4],dim=1))
        x_gauss = x[:,feature_type_dict['real_data']]
        x_bernoulli = (torch.sign(x[:,feature_type_dict['binary_data']])+1)/2
        # print(x_bernoulli)
        # print(p_bernoulli)
        #! LOSS
        BC = F.binary_cross_entropy(p_bernoulli, x_bernoulli, reduction='sum')
        RE, _ = self.decoder.log_probas(x_gauss,mu_gauss, std_gauss)  
        #KL = torch.distributions.kl_divergence(qz, pz_gauss).sum(dim=1)
        #KL = -self.kl_div([None,zs[-1]],[None,mu],[None,std], self.prior, self.encoder, 'prpo')
        KL = (self.log_normal_diag(zs[-1], torch.sqrt(1. - self.beta) * zs[-1], torch.log(self.beta)) - self.log_standard_normal(zs[-1])).sum(-1)

        for i in range(len(mus)):
            KL_i = (self.log_normal_diag(zs[i], torch.sqrt(1. - self.beta) * zs[i], torch.log(self.beta)) - self.log_normal_diag(zs[i], mus[i], torch.log(stds[i]))).sum(-1)

            KL = KL + KL_i
        MC = sum(F.cross_entropy(p_categorical[:,start_p:end_p], x[:,start_x:end_x].argmax(dim=1), reduction='sum') for (start_x, end_x), (start_p, end_p) in zip(feature_type_dict['categorical_one_hot'], feature_type_dict['categorical_only']))

        beta_param = 0.01
        gamma_param = 0.01
        
        return -torch.mean(RE) + torch.mean(KL) + beta_param*BC + gamma_param*MC


    def count_params(self):
        return sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)

    def kl_div(self,z,mu,std, dist_1, dist_2, type):
        kl_part_1 = None
        kl_part_2 = None
        # prior and posterior
        if type == 'prpo':
            kl_part_1 = dist_1.log_probas(z[1])
            #print(kl_part_1)
        # two posteriors
        elif type == 'popo':
            kl_part_1 = dist_1.log_probas(z[0], mu[0], std[0])[0]
        kl_part_2 = dist_2.log_probas(z[1], mu[1], std[1])[0] 
        #print(kl_part_2)
        #exit()
        KL = kl_part_1 - kl_part_2
        return KL
    def linear_beta_scheduler(self, timesteps):
        beta_start = 1e-4
        beta_end = 0.02
        return torch.linspace(beta_start, beta_end, timesteps, requires_grad=False)

    def reparameterization_gaussian_diffusion(self, x, i):
        return torch.sqrt(1. - self.beta) * x + torch.sqrt(self.beta) * torch.randn_like(x)
    
    def log_normal_diag(self,x, mu, log_var, reduction=None, dim=None):
        PI = torch.from_numpy(np.asarray(np.pi))
        log_p = -0.5 * torch.log(2. * PI) - 0.5 * log_var - 0.5 * torch.exp(-log_var) * (x - mu)**2.
        if reduction == 'avg':
            return torch.mean(log_p, dim)
        elif reduction == 'sum':
            return torch.sum(log_p, dim)
        else:
            return log_p

    def log_standard_normal(self, x, reduction=None, dim=None):
        PI = torch.from_numpy(np.asarray(np.pi))
        log_p = -0.5 * torch.log(2. * PI) - 0.5 * x**2.
        if reduction == 'avg':
            return torch.mean(log_p, dim)
        elif reduction == 'sum':
            return torch.sum(log_p, dim)
        else:
            return log_p