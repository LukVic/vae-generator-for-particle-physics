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
            #if relu[idx] != 0: layers.append(nn.Sigmoid())
            #if drop[idx] != 0: layers.append(nn.Dropout(drop[idx]))
        
        self.body = nn.Sequential(*layers)


    def forward(self, z):
        xhat = self.body(z)
        xhat_gauss = xhat[:, :-1]
        xhat_bernoulli = xhat[:,-1]
        # print(xhat_gauss.shape)
        # print(self.input_size-1)
        xhat_gauss_mu, xhat_gauss_sigma = torch.split(xhat_gauss, self.input_size - 1, dim=1)
        xhat_bernoulli = torch.sigmoid(xhat_bernoulli)
        return xhat_gauss_mu, xhat_gauss_sigma, xhat_bernoulli
        
        #xhat[:,-1] = torch.bernoulli(torch.sigmoid(xhat[:,-1]))
        
        #return xhat
        # return xhat_gauss, xhat_bernoulli


class VAE(nn.Module):
    def __init__(self, zdim, device, input_size, config:dict):
        super(VAE, self).__init__()
        self.device = device
        self.zdim = zdim
        self.input_size = input_size
        self.config = config
        
        self.encoder = Encoder(self.zdim, self.input_size, self.config).to(self.device)
        self.decoder = Decoder(self.zdim, self.input_size, self.config).to(self.device)

    def forward(self, x):
        #! ENCODER
        x = x.to(self.device)
        mu, sigma = self.encoder(x.view(-1, self.input_size))
        std = torch.exp(sigma)  
        qz_gauss = torch.distributions.Normal(mu, std)
        z = qz_gauss.rsample()
        pz_gauss = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        #! DECODER
        mu_gauss, sigma_gauss, p_bernoulli = self.decoder(z)
        xhat_gauss = torch.cat((mu_gauss, sigma_gauss), dim=1)
        xhat = torch.cat((xhat_gauss, p_bernoulli.view(-1,1)), dim=1)
        return xhat, pz_gauss, qz_gauss

    def count_params(self):
        return sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)


    # Computes reconstruction loss
    def recon(self, x_hat, x):
        xhat_gauss_mu, xhat_gauss_sigma = torch.split(x_hat, x.shape[1], dim=1) 
        std = torch.exp(xhat_gauss_sigma)
        pxz = Normal(xhat_gauss_mu,  std)
        E_log_pxz = -pxz.log_prob(x.to(self.device)).sum(dim=1)
        return E_log_pxz


    def loss_function(self, x, x_hat, pz, qz):
        pz_gauss = pz
        qz_gauss = qz
        
        x_gauss = x[:, :-1]
        x_bernoulli = x[:, -1].to(torch.float32).to(self.device)
        
        x_hat_gauss = x_hat[:, :-1]
        x_hat_bernoulli = x_hat[:, -1].to(torch.float32).to(self.device)
        
        x_bernoulli = torch.sigmoid(x_bernoulli)
        
        REC_G = self.recon(x_hat_gauss, x_gauss)
        KLD_G = torch.distributions.kl_divergence(qz, pz_gauss).sum(dim=1)
        BCE_B = F.binary_cross_entropy(x_bernoulli, x_hat_bernoulli, reduction='sum')
        
        beta = 1.0

        #return torch.mean(REC_G + beta*KLD_G) 
        return torch.mean(REC_G + beta*(0.1*BCE_B + KLD_G))