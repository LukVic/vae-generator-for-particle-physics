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
                layers.append(nn.Linear(arch[idx][0], self.zdim*2 + 1))
                #init.xavier_uniform_(layers[-1].weight)
            else: 
                layers.append(nn.Linear(arch[idx][0],arch[idx][1]))
                #init.xavier_uniform_(layers[-1].weight)
            
            if bNorm[idx] != 0: layers.append(nn.BatchNorm1d(num_features=bNorm[idx]))
            if relu[idx] != 0: layers.append(nn.ReLU())
            if drop[idx] != 0: layers.append(nn.Dropout(drop[idx]))
        
        self.body = nn.Sequential(*layers)
        

    def forward(self, x):
        scores = self.body(x)
        # print(f"SCORES {scores.shape}")
        gauss, bern = torch.split(scores, 2*self.zdim, dim=1)
        p = torch.sigmoid(bern)
        # print(f"GAUSS {gauss.shape}")
        # print(f"BERN {bern.shape}")
        mu, sigma = torch.split(gauss, self.zdim, dim=1)
        # print(f"MU {mu.shape}")
        # print(f"SIGMA {sigma.shape}")
        return mu, sigma, p

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
                layers.append(nn.Linear(self.zdim + 1, arch[idx][0]))
                #init.xavier_uniform_(layers[-1].weight)
            elif idx == layer_num - 1: 
                layers.append(nn.Linear(arch[idx][0], self.input_size))
                #init.xavier_uniform_(layers[-1].weight)
            else: 
                layers.append(nn.Linear(arch[idx][0],arch[idx][1]))
                #init.xavier_uniform_(layers[-1].weight)
            
            if bNorm[idx] != 0: layers.append(nn.BatchNorm1d(num_features=bNorm[idx]))
            if relu[idx] != 0: layers.append(nn.ReLU())
            if drop[idx] != 0: layers.append(nn.Dropout(drop[idx]))
        
        self.body = nn.Sequential(*layers)


    def forward(self, z):
        xhat = self.body(z)
        xhat[:, 7] = torch.sigmoid(xhat[:, 7])
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
        mu, sigma, p = self.encoder(x.view(-1, self.input_size))
        
        std = torch.exp(sigma)  
        qz_gauss = torch.distributions.Normal(mu, std)
        z_gauss = qz_gauss.rsample()
        z_bernoulli = torch.bernoulli(p)
        
        #print(f"P {p}")
        
        z = torch.cat([z_gauss, z_bernoulli], dim=1)
        pz_gauss = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(sigma))
        pz_bernoulli = torch.distributions.Bernoulli(p)
        xhat = self.decoder(z)
        #?logqz = qz.log_prob(z)
        return xhat, [pz_gauss, pz_bernoulli], [qz_gauss, z_bernoulli]

    def count_params(self):
        return sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)


    # Computes reconstruction loss
    def recon(self, x_hat, x):
        
        pxz = Normal(x_hat, torch.ones_like(x_hat))
        E_log_pxz = -pxz.log_prob(x.to(self.device)).sum(dim=1)

        return E_log_pxz

    def loss_function(self, x, x_hat, pz, qz):
        pz_gauss = pz[0]
        pz_bernoulli = pz[1]
        
        qz_gauss = qz[0]
        z_bernoulli = qz[1]
        
        x_gauss = x[:, torch.arange(x.size(1)) != 7]
        x_bernoulli = x[:, 7].to(torch.float32).to(self.device)
        
        x_hat_gauss = x_hat[:, torch.arange(x_hat.size(1)) != 7]
        x_hat_bernoulli = x_hat[:, 7].to(torch.float32).to(self.device)
        
        #for val in x_bernoulli: print(val)
        #exit()
        # print(x_bernoulli)
        
        #x_gauss = x
        #x_hat_gauss = x_hat
        
        
        REC_G = self.recon(x_hat_gauss, x_gauss)
        KLD_G = torch.distributions.kl_divergence(qz_gauss, pz_gauss).sum(dim=1)
        
        x_bernoulli = torch.sigmoid(x_bernoulli)
        #for val in x_bernoulli: print(val)
        #exit()
        BCE_B = F.binary_cross_entropy(x_bernoulli, x_hat_bernoulli, reduction='sum')
        
        beta = 0.1

        return torch.mean(REC_G + 0.1*BCE_B + beta*(KLD_G)) 
        #return torch.mean(REC_G + beta*(BCE_B + KLD_G))
    
