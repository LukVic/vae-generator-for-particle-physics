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
        self.conf_general_encode = config["generate"]["encoder"]
        
        layer_num = self.conf_general_encode["layer_num"]
        arch = self.conf_general_encode["architecture"]
        bNorm = self.conf_general_encode["batchNorm"]
        relu = self.conf_general_encode["relu"]
        drop = self.conf_general_encode["dropout"]
        
        layers = []
        
        def res_block():
            ...
        
        
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
            
            #if bNorm[idx] != 0: layers.append(nn.BatchNorm1d(num_features=bNorm[idx]))
            if bNorm[idx] != 0:layers.append(nn.LayerNorm(normalized_shape=bNorm[idx]))
            #if bNorm[idx] != 0:layers.append(nn.InstanceNorm1d(num_features=bNorm[idx]))
            #if relu[idx] != 0: layers.append(nn.ReLU())
            if relu[idx] != 0: layers.append(nn.GELU())
            #if drop[idx] != 0: layers.append(nn.Dropout(drop[idx]))
        
        self.body = nn.Sequential(*layers)
        
    def sample(self, mu, std, delta_mu, delta_std):
        
        qz_gauss= torch.distributions.Normal(mu + delta_mu, std * delta_std)
        
        # ones = torch.ones(delta_std.shape).to(self.device)
        # sigma_new_2 = (std * delta_std.pow(2))/(ones + delta_std.pow(2))
        # mu_new_2 = (mu * delta_std.pow(2) + delta_mu.pow(2) * std)/(ones + delta_std.pow(2))
        
        # qz_gauss = torch.distributions.Normal(mu_new_2, sigma_new_2)
        
        z = qz_gauss.rsample()
        return z, qz_gauss

    def forward(self, x):
        scores = self.body(x)
        mu, sigma = torch.split(scores, self.zdim, dim=1)
        return mu, sigma

class Decoder(nn.Module):
    def __init__(self, zdim, input_size, config:dict):
        super(Decoder, self).__init__()
        self.zdim = zdim
        self.input_size = input_size
        self.conf_general_encode = config["generate"]["encoder"]
        
        layer_num = self.conf_general_encode["layer_num"]
        arch = self.conf_general_encode["architecture"]
        bNorm = self.conf_general_encode["batchNorm"]
        relu = self.conf_general_encode["relu"]
        drop = self.conf_general_encode["dropout"]
        
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
            
            #if bNorm[idx] != 0: layers.append(nn.BatchNorm1d(num_features=bNorm[idx]))
            if bNorm[idx] != 0:layers.append(nn.LayerNorm(normalized_shape=bNorm[idx]))
            #if bNorm[idx] != 0:layers.append(nn.InstanceNorm1d(num_features=bNorm[idx]))
            #if relu[idx] != 0: layers.append(nn.ReLU())
            if relu[idx] != 0: layers.append(nn.GELU())
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
        self.conf_general_encode = config["generate"]["encoder"]
        
        layer_num = self.conf_general_encode["layer_num"]
        arch = self.conf_general_encode["architecture"]
        bNorm = self.conf_general_encode["batchNorm"]
        relu = self.conf_general_encode["relu"]
        drop = self.conf_general_encode["dropout"]
        
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
            
            #if bNorm[idx] != 0: layers.append(nn.BatchNorm1d(num_features=bNorm[idx]))
            if bNorm[idx] != 0:layers.append(nn.LayerNorm(normalized_shape=bNorm[idx]))
            #if bNorm[idx] != 0:layers.append(nn.InstanceNorm1d(num_features=bNorm[idx]))
            #if relu[idx] != 0: layers.append(nn.ReLU())
            if relu[idx] != 0: layers.append(nn.GELU())
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
        self.encoder_1 = Encoder(self.zdim, self.r_dim, self.config).to(self.device) #zdim*2
        self.encoder_2 = Encoder(self.zdim, self.r_dim, self.config).to(self.device)
        self.encoder_3 = Encoder(self.zdim, self.zdim, self.config).to(self.device) #zdim*2
        self.decoder = Decoder(self.zdim, self.input_size, self.config).to(self.device) #zdim*2

    def forward(self, x):
        #! ENCODER
        x = x.to(self.device)
        
        
        #!DETERMINISTIC PATH
        r_1 = self.deterministic_encoder_1(x)
        r_2 = self.deterministic_encoder_2(r_1)
        
        delta_mu_1, delta_sigma_1 = self.encoder_1(r_1.view(-1, self.r_dim))
        #delta_sigma_1 = F.hardtanh(delta_sigma_1, -7., 2.)
        delta_mu_2, delta_sigma_2 = self.encoder_2(r_2.view(-1, self.r_dim))
        #delta_sigma_2 = F.hardtanh(delta_sigma_2, -7., 2.)
        delta_std_1 = torch.exp(delta_sigma_1)
        delta_std_2 = torch.exp(delta_sigma_2)

        
        z_2, qz_gauss_2 = self.encoder_2.sample(0,1,delta_mu_2,delta_std_2)
        
        
        mu_1, sigma_1 = self.encoder_3(z_2.view(-1, self.zdim))
        std_1 = torch.exp(sigma_1)
        
        z_1, qz_gauss_1 = self.encoder_1.sample(mu_1,std_1,delta_mu_1,delta_std_1)
        
        pz_2_gauss = torch.distributions.Normal(torch.zeros_like(delta_mu_2), torch.ones_like(delta_std_2))
        pz_1_gauss = torch.distributions.Normal(mu_1, std_1)
        
        #! DECODER
        mu_gauss, sigma_gauss, p_bernoulli = self.decoder(z_1)
        xhat_gauss = torch.cat((mu_gauss, sigma_gauss), dim=1)
        x_hat = torch.cat((xhat_gauss, p_bernoulli.view(-1,1)), dim=1)
        
        #! LOSS 
        x_gauss = x[:, :-1]
        x_bernoulli = x[:, -1].to(torch.float32).to(self.device)
        
        x_hat_gauss = x_hat[:, :-1]
        x_hat_bernoulli = x_hat[:, -1].to(torch.float32).to(self.device)
        
        x_bernoulli = torch.sigmoid(x_bernoulli)
        
        REC_G = self.recon(x_hat_gauss, x_gauss)
        KLD_G_1 = torch.distributions.kl_divergence(qz_gauss_1, pz_1_gauss).sum(dim=1)
        KLD_G_2 = torch.distributions.kl_divergence(qz_gauss_2, pz_2_gauss).sum(dim=1)
        BCE_B = F.binary_cross_entropy(x_bernoulli, x_hat_bernoulli, reduction='sum')
        
        beta = 1.0

        return torch.mean(REC_G + beta*(KLD_G_1 + KLD_G_2)) + 0.001*BCE_B 
        #return torch.mean(REC_G + beta*(BCE_B + KLD_G))


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

