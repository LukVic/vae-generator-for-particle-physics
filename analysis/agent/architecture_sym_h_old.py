import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import torch.nn.init as init
import optuna

class Encoder(nn.Module):
    def __init__(self, zdim, input_size, config:dict):
        super(Encoder, self).__init__()
        self.zdim = zdim
        self.input_size = input_size
        self.config = config
        
        layer_num = config["encoder"]["layer_num"]
        arch = config["encoder"]["architecture"]
        bNorm = config["encoder"]["batchNorm"]
        relu = config["encoder"]["relu"]
        drop = config["encoder"]["dropout"]
        
        layers = []
        for idx in range(layer_num):
            if idx == 0: 
                layers.append(nn.Linear(self.input_size, arch[idx][0]))
                #init.xavier_uniform_(layers[-1].weight)
            elif idx == layer_num - 1: 
                layers.append(nn.Linear(arch[idx][0], self.zdim*2))
                #init.xavier_uniform_(layers[-1].weight)
            else: 
                layers.append(nn.Linear(arch[idx][0],arch[idx][1]))
                #init.xavier_uniform_(layers[-1].weight)
            
            if bNorm[idx] != 0: layers.append(nn.BatchNorm1d(num_features=bNorm[idx]))
            #if bNorm[idx] != 0:layers.append(nn.LayerNorm(normalized_shape=bNorm[idx]))
            #if bNorm[idx] != 0:layers.append(nn.InstanceNorm1d(num_features=bNorm[idx]))
            if relu[idx] != 0: layers.append(nn.ReLU())
            #if relu[idx] != 0: layers.append(nn.GELU())
            #if drop[idx] != 0: layers.append(nn.Dropout(drop[idx]))
        
        self.body = nn.Sequential(*layers)
        

    def forward(self, x):
        scores = self.body(x)
        mu, sigma = torch.split(scores, self.zdim, dim=1)
        return mu, sigma

class Decoder(nn.Module):
    def __init__(self, zdim, input_size, config:dict):
        super(Decoder, self).__init__()
        self.zdim = zdim
        self.input_size = input_size
        self.config = config
        
        layer_num = config["decoder"]["layer_num"]
        arch = config["decoder"]["architecture"]
        bNorm = config["decoder"]["batchNorm"]
        relu = config["decoder"]["relu"]
        drop = config["decoder"]["dropout"]
        
        layers = []
        for idx in range(layer_num):
            if idx == 0: 
                layers.append(nn.Linear(self.zdim, arch[idx][0]))
                #init.xavier_uniform_(layers[-1].weight)
            elif idx == layer_num - 1: 
                layers.append(nn.Linear(arch[idx][0], self.input_size*2-1))
                #init.xavier_uniform_(layers[-1].weight)
            else: 
                layers.append(nn.Linear(arch[idx][0],arch[idx][1]))
                #init.xavier_uniform_(layers[-1].weight)
            
            if bNorm[idx] != 0: layers.append(nn.BatchNorm1d(num_features=bNorm[idx]))
            #if bNorm[idx] != 0:layers.append(nn.LayerNorm(normalized_shape=bNorm[idx]))
            #if bNorm[idx] != 0:layers.append(nn.InstanceNorm1d(num_features=bNorm[idx]))
            if relu[idx] != 0: layers.append(nn.ReLU())
            #if relu[idx] != 0: layers.append(nn.GELU())
            #if relu[idx] != 0: layers.append(nn.Sigmoid())
            #if drop[idx] != 0: layers.append(nn.Dropout(drop[idx]))
        
        self.body = nn.Sequential(*layers)

    def forward(self, z):
        xhat = self.body(z)
        xhat_gauss = xhat[:, :-1]
        xhat_bernoulli = xhat[:,-1]
        xhat_gauss_mu, xhat_gauss_sigma = torch.split(xhat_gauss, self.input_size - 1, dim=1)
        xhat_bernoulli = torch.sigmoid(xhat_bernoulli)
        return xhat_gauss_mu, xhat_gauss_sigma, xhat_bernoulli
        


class Deterministic_encoder(nn.Module):
    def __init__(self, input_size, r_size ,config:dict):
        super(Deterministic_encoder, self).__init__()
        self.r_size = r_size
        self.input_size = input_size
        self.config = config
        
        layer_num = config["decoder"]["layer_num"]
        arch = config["decoder"]["architecture"]
        bNorm = config["decoder"]["batchNorm"]
        relu = config["decoder"]["relu"]
        drop = config["decoder"]["dropout"]
        
        layers = []
        for idx in range(layer_num):
            if idx == 0: 
                layers.append(nn.Linear(self.input_size, arch[idx][0]))
                #init.xavier_uniform_(layers[-1].weight)
            elif idx == layer_num - 1: 
                layers.append(nn.Linear(arch[idx][0], arch[idx][0]))
                #init.xavier_uniform_(layers[-1].weight)
            else: 
                layers.append(nn.Linear(arch[idx][0],self.r_size))
                #init.xavier_uniform_(layers[-1].weight)
            
            if bNorm[idx] != 0: layers.append(nn.BatchNorm1d(num_features=bNorm[idx]))
            #if bNorm[idx] != 0:layers.append(nn.LayerNorm(normalized_shape=bNorm[idx]))
            #if bNorm[idx] != 0:layers.append(nn.InstanceNorm1d(num_features=bNorm[idx]))
            if relu[idx] != 0: layers.append(nn.ReLU())
            #if relu[idx] != 0: layers.append(nn.GELU())
            #if relu[idx] != 0: layers.append(nn.Sigmoid())
            #if drop[idx] != 0: layers.append(nn.Dropout(drop[idx]))
        
        self.body = nn.Sequential(*layers)

    def forward(self, x):
        r = self.body(x)
        return r



class VAE(nn.Module):
    def __init__(self, zdim, device, input_size, config:dict):
        super(VAE, self).__init__()
        self.device = device
        self.zdim = zdim
        self.input_size = input_size
        self.config = config
        self.r_dim = config['general']['r_size']

         
        self.deterministic_encoder_1 = Deterministic_encoder(self.input_size, self.r_dim, self.config).to(self.device)
        self.deterministic_encoder_2 = Deterministic_encoder(self.r_dim, self.r_dim, self.config).to(self.device)    
        self.encoder_1 = Encoder(self.zdim, self.r_dim, self.config).to(self.device)
        self.encoder_2 = Encoder(self.zdim, self.r_dim, self.config).to(self.device)
        self.encoder_3 = Encoder(self.zdim, self.zdim, self.config).to(self.device)
        self.decoder = Decoder(self.zdim, self.input_size, self.config).to(self.device)

    def forward(self, sample, step):
        #! THE FIRST STEP
        if step == 1:
            x = sample.to(self.device)
            r_1 = self.deterministic_encoder_1(x.view(-1, self.input_size))
            r_2 = self.deterministic_encoder_2(r_1.view(-1, self.r_dim))
            delta_mu_1, delta_sigma_1 = self.encoder_1(r_1.view(-1, self.r_dim))
            delta_mu_2, delta_sigma_2 = self.encoder_2(r_2.view(-1, self.r_dim))
            delta_std_1 = torch.exp(delta_sigma_1)
            delta_std_2 = torch.exp(delta_sigma_2)  
            
            ones = torch.ones(delta_std_2.shape).to(self.device)
            std_new_2 = (1 * delta_std_2.pow(2))/(ones + delta_std_2.pow(2))
            mu_new_2 = (0 * delta_std_2.pow(2) + delta_mu_2 * 1)/(ones + delta_std_2.pow(2))
            
            # std_new_2 = delta_std_2 + 0
            # mu_new_2 = delta_mu_2 + 0
            
            qz_gauss_2 = torch.distributions.Normal(mu_new_2, std_new_2)
            
            
            z_2 = qz_gauss_2.sample()
            mu_1, sigma_1 = self.encoder_3(z_2.view(-1, self.zdim))
            std_1 = torch.exp(sigma_1)
            
            std_new_1 = (delta_std_1.pow(2) * std_1.pow(2))/(delta_std_1.pow(2) + std_1.pow(2))
            mu_new_1 = (delta_mu_1 * std_1.pow(2) + mu_1 * delta_std_1.pow(2))/(delta_std_1.pow(2) + std_1.pow(2))
            
            # std_new_1 = delta_std_1 + std_1
            # mu_new_1 = delta_mu_1 + mu_1
            
            qz_gauss_1 = torch.distributions.Normal(mu_new_1, std_new_1)
            z_1 = qz_gauss_1.sample()
            
            return z_1, z_2, x
        
        #! THE SECOND STEP
        elif step == 2:
            z_2 = sample.to(self.device)
            mu_1, sigma_1 = self.encoder_3(z_2.view(-1, self.zdim))
            std_1 = torch.exp(sigma_1)
            
            pz1z2 = torch.distributions.Normal(mu_1, std_1)
            z_1 = pz1z2.sample()
            
            mu_gauss, sigma_gauss, p_bernoulli = self.decoder(z_1)
            std = torch.exp(sigma_gauss)
            pxz_gauss = torch.distributions.Normal(mu_gauss, std)
            x_gauss = pxz_gauss.sample()
            x_bernoulli = torch.bernoulli(p_bernoulli)
            x = torch.cat((x_gauss, x_bernoulli.view(-1,1)), dim=1)
            return z_1, z_2, x

    def count_params(self):
        return sum(p.numel() for p in self.encoder_1.parameters() if p.requires_grad)


    # Computes reconstruction loss
    def recon(self, x_hat, x):
        # print(f"x_hat: {x_hat.shape}")
        # print(f"x: {x.shape}")
        xhat_gauss_mu, xhat_gauss_sigma = torch.split(x_hat, x.shape[1], dim=1) 
        std = torch.exp(xhat_gauss_sigma)
        pxz = Normal(xhat_gauss_mu,  std)
        E_log_pxz = -pxz.log_prob(x.to(self.device)).sum(dim=1)
        return E_log_pxz


    def loss_function(self, variables, step):
        z_1 = variables[0]
        z_2 = variables[1]
        x = variables[2]
        x_gauss = x[:, :-1]
        

        
        if step == 1:
            
            #pz_2 = torch.distributions.Normal(torch.zeros_like(z_2), torch.ones_like(z_2))
            #log_pz_2 = -pz_2.log_prob(z_2.to(self.device)).sum(dim=1)
            
            mu_z_1, sigma_z_1 = self.encoder_3(z_2)
            std_z_1 = torch.exp(sigma_z_1)
            pz_1z_2 = torch.distributions.Normal(mu_z_1, std_z_1)
            
            mu_x, sigma_x, p_x = self.decoder(z_1)
            std_x = torch.exp(sigma_x)
            pxz_1 = torch.distributions.Normal(mu_x, std_x)
            
            log_px_pz_1 = -pxz_1.log_prob(x_gauss.to(self.device)).sum(dim=1)
            log_pz_1_pz_2 = -pz_1z_2.log_prob(z_1.to(self.device)).sum(dim=1)
            
                
            E_log_probs = torch.mean(log_px_pz_1) + torch.mean(log_pz_1_pz_2) #+ torch.mean(log_pz_2)
            LOSS_G = E_log_probs
            
            return LOSS_G   
            #return torch.mean(LOSS_G)
        
        elif step == 2:
            
            r_1 = self.deterministic_encoder_1(x.view(-1, self.input_size))
            r_2 = self.deterministic_encoder_2(r_1.view(-1, self.r_dim))
            delta_mu_1, delta_sigma_1 = self.encoder_1(r_1.view(-1, self.r_dim))
            delta_mu_2, delta_sigma_2 = self.encoder_2(r_2.view(-1, self.r_dim))
            delta_std_1 = torch.exp(delta_sigma_1)
            delta_std_2 = torch.exp(delta_sigma_2)
            
            ones = torch.ones(delta_std_2.shape).to(self.device)
            std_new_2 = (1 * delta_std_2.pow(2))/(ones + delta_std_2.pow(2))
            mu_new_2 = (delta_std_2.pow(2) * 0 + delta_mu_2 * 1)/(ones + delta_std_2.pow(2))
            
            # std_new_2 = delta_std_2 + 0
            # mu_new_2 = delta_mu_2 + 0
            
            qz_2_x = torch.distributions.Normal(mu_new_2, std_new_2)
            log_pz_2_px = -qz_2_x.log_prob(z_2.to(self.device)).sum(dim=1)
            
            mu_z_1, sigma_z_1 = self.encoder_3(z_2)
            std_z_1 = torch.exp(sigma_z_1)
            
            mu_z_1 = mu_z_1.detach().clone()
            std_z_1 = std_z_1.detach().clone()
            
            std_new_1 = (delta_std_1.pow(2) * std_z_1.pow(2))/(delta_std_1.pow(2) + std_z_1.pow(2))
            mu_new_1 = (delta_mu_1 * std_z_1.pow(2) + delta_std_1.pow(2) * mu_z_1)/(std_z_1.pow(2) + delta_std_1.pow(2))
            
            #std_new_1 = std_z_1 + delta_std_1
            #mu_new_1 = mu_z_1 + delta_mu_1
            
            qz_1z_2_x = torch.distributions.Normal(mu_new_1, std_new_1)
            log_pz_1pz_2_px = -qz_1z_2_x.log_prob(z_1.to(self.device)).sum(dim=1)

            #E_log_probs = log_px_pz_1 + log_pz_1_pz_2 + log_pz_2 + log_pz_2_px #+ log_pz_1pz_2_px
            E_log_probs = torch.mean(log_pz_2_px) + torch.mean(log_pz_1pz_2_px)
            #E_log_probs = torch.mean(log_px_pz_1)  + torch.mean(log_pz_2) #+ log_pz_1_pz_2
            LOSS_G = E_log_probs
            
            return LOSS_G
            #return torch.mean(LOSS_G)