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
            
            #if bNorm[idx] != 0: layers.append(nn.BatchNorm1d(num_features=bNorm[idx]))
            if bNorm[idx] != 0:layers.append(nn.LayerNorm(normalized_shape=bNorm[idx]))
            if relu[idx] != 0: layers.append(nn.ReLU())
            if drop[idx] != 0: layers.append(nn.Dropout(drop[idx]))
        
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
            
            #if bNorm[idx] != 0: layers.append(nn.BatchNorm1d(num_features=bNorm[idx]))
            if bNorm[idx] != 0:layers.append(nn.LayerNorm(normalized_shape=bNorm[idx]))
            if relu[idx] != 0: layers.append(nn.ReLU())
            if drop[idx] != 0: layers.append(nn.Dropout(drop[idx]))
        
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
        

class VAE(nn.Module):
    def __init__(self, zdim, device, input_size, config:dict):
        super(VAE, self).__init__()
        self.device = device
        self.zdim = zdim
        self.input_size = input_size
        self.config = config
        
        self.encoder = Encoder(self.zdim, self.input_size, self.config).to(self.device)
        self.decoder = Decoder(self.zdim, self.input_size, self.config).to(self.device)

    def forward(self, sample, step):
        if step == 1:
            #! THE FIRST STEP
            x = sample.to(self.device)
            mu, sigma = self.encoder(x.view(-1, self.input_size))
            std = torch.exp(sigma)  
            qzx_gauss = torch.distributions.Normal(mu, std)
            z = qzx_gauss.sample()
            decoder_out = self.decoder(z)
            return decoder_out
            
        
        elif step == 2:
            #! THE SECOND STEP
            z = sample.to(self.device)
            mu_gauss, sigma_gauss, p_bernoulli = self.decoder(z)
            std = torch.exp(sigma_gauss)
            # pxz_gauss = Normal(mu_gauss, torch.ones_like(std))
            pxz_gauss = torch.distributions.Normal(mu_gauss, std)
            x_gauss = pxz_gauss.sample()
            x_bernoulli = torch.bernoulli(p_bernoulli)
            x = torch.cat((x_gauss, x_bernoulli.view(-1,1)), dim=1)
            encoder_out = self.encoder(x.view(-1, self.input_size))
            return encoder_out

    def count_params(self):
        return sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)

    def loss_function(self, x, distr_out, step):
        if step == 1:
            
            #! DECODER OUTPUT
            mu = distr_out[0]
            sigma = distr_out[1]
            p = distr_out[2]
            
            #! CONTINUOUS AND DISCRETE SEPARATION
            x_gauss = x[:, :-1]
            x_bernoulli = x[:, -1].to(torch.float32).to(self.device)
            
            #! P(X|Z) CONSTRUCTION
            std = torch.exp(sigma)
            #pxz = Normal(mu, torch.ones_like(std))
            pxz = Normal(mu, std)
            E_log_pxz = -pxz.log_prob(x_gauss.to(self.device)).sum(dim=1)
            
            #! DISCRETE PART
            x_bernoulli = torch.sigmoid(x_bernoulli)
            
            #! COMPUTE THE LOSS
            LOSS_G = E_log_pxz
            LOSS_B = F.binary_cross_entropy(x_bernoulli, p, reduction="sum")
            
            #return torch.mean(LOSS_G + 0.1*LOSS_B)
            return torch.mean(LOSS_G) 
        
        elif step == 2:
            
            #! ENCODER OUTPUT
            mu = distr_out[0]
            sigma = distr_out[1]
            
            #! Q(Z|X) CONSTRUCTION
            std = torch.exp(sigma)
            qzx = Normal(mu, std)
            E_log_qzx = -qzx.log_prob(x.to(self.device)).sum(dim=1)
            
            #! COMPUTE THE LOSS
            LOSS = E_log_qzx
            
            return torch.mean(LOSS)
        
        
