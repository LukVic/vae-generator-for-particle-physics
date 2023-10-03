import torch
import torch.nn as nn
from torch.distributions.normal import Normal
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
            if idx == 0: layers.append(nn.Linear(self.input_size, arch[idx][0]))
            elif idx == layer_num - 1: layers.append(nn.Linear(arch[idx][0], self.zdim*2))
            else: layers.append(nn.Linear(arch[idx][0],arch[idx][1]))
            
            if bNorm[idx] != 0: layers.append(nn.BatchNorm1d(num_features=bNorm[idx]))
            if relu[idx] != 0: layers.append(nn.ReLU())
            if drop[idx] != 0: layers.append(nn.Dropout(drop[idx]))
        
        self.body = nn.Sequential(*layers)
        
        # self.body = nn.Sequential(
        #     nn.Linear(input_size, 128),
        #     nn.BatchNorm1d(num_features=128),
        #     nn.ReLU(),
        #     nn.Dropout(0.1),
        #     nn.Linear(128, 1024),
        #     nn.BatchNorm1d(num_features=1024),
        #     nn.ReLU(),
        #     nn.Dropout(0.1),
        #     nn.Linear(1024, 1024),
        #     nn.BatchNorm1d(num_features=1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, self.zdim * 2)
        #     #nn.Linear(56, self.zdim*2)
        #)

    def forward(self, x):
        scores = self.body(x)
        #print(scores)
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
            if idx == 0: layers.append(nn.Linear(self.zdim, arch[idx][0]))
            elif idx == layer_num - 1: layers.append(nn.Linear(arch[idx][0], self.input_size))
            else: layers.append(nn.Linear(arch[idx][0],arch[idx][1]))
            
            if bNorm[idx] != 0: layers.append(nn.BatchNorm1d(num_features=bNorm[idx]))
            if relu[idx] != 0: layers.append(nn.ReLU())
            if drop[idx] != 0: layers.append(nn.Dropout(drop[idx]))
        
        self.body = nn.Sequential(*layers)
        
        
        
        # self.body = nn.Sequential(
        #     nn.Linear(self.zdim, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 1024),
        #     nn.BatchNorm1d(num_features=1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, 128),
        #     nn.BatchNorm1d(num_features=128),
        #     nn.ReLU(),
        #     nn.Linear(128,input_size),
        #     nn.BatchNorm1d(num_features=input_size),
            #nn.Sigmoid()

            # nn.Linear(self.zdim,56),
            # nn.Sigmoid()
        # )

    def forward(self, z):
        xhat = self.body(z)
        return xhat

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
        x = x.to(self.device)
        mu, sigma = self.encoder(x.view(-1, self.input_size))
        std = torch.exp(sigma)  
        
        qz = torch.distributions.Normal(mu, std)
        pz = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(sigma))
        
        z = qz.rsample()
        xhat = self.decoder(z)
        #?logqz = qz.log_prob(z)

        return xhat, pz, qz

    def count_params(self):
        return sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)


    # Computes reconstruction loss
    def recon(self, x_hat, x):
        
        pxz = Normal(x_hat, torch.ones_like(x_hat))
        E_log_pxz = -pxz.log_prob(x.to(self.device)).sum(dim=1)

        return E_log_pxz

    def loss_function(self, x, x_hat, pz, qz):
        REC = self.recon(x_hat, x)
        #eps = 0.00000001
        #BCE = nn.functional.binary_cross_entropy(x_hat, x.view(-1,56), reduction='sum')
        #BCE = -torch.sum(x * torch.log(x_hat + eps) + (1 - x) * torch.log(1 - x_hat + eps))
        
        #KLD = torch.distributions.kl_divergence(qz, pz).sum()
        KLD = torch.distributions.kl_divergence(qz, pz).sum(dim=1)
        
        # if i == 0:
        #     beta = 0.0
        # elif i == 1:
        #     beta = 0.1
        # else: beta = 0.3
        #if i % 2 == 0: beta = 0.2
        #else: beta = 0.7

        beta = 0.1

        return torch.mean(REC + beta*KLD)
    
