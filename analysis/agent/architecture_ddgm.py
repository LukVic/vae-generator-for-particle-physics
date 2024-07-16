import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import torch.nn.init as init
import numpy as np
import matplotlib.pyplot as plt

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
            #if bNorm[idx] != 0:layers.append(nn.InstanceNorm1d(num_features=bNorm[idx]))
            #if relu[idx] != 0: layers.append(nn.ReLU())
            if relu[idx] != 0: layers.append(nn.GELU())
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
        

class DDGM(nn.Module):
    def __init__(self, zdim, device, input_size, config:dict):
        super(DDGM, self).__init__()
        self.device = device
        self.zdim = zdim
        self.input_size = input_size
        self.config = config
        self.r_dim = config['general']['r_size']
        self.beta = config['general']['beta']
        self.dif_depth = config['general']['dif_depth']
        
        self.encoders = nn.ModuleList()
        
        for _ in range(self.dif_depth):
            self.encoders.append(Encoder(self.zdim, self.zdim, self.config).to(self.device))
        
        self.decoder = Decoder(self.zdim, self.input_size, self.config).to(self.device) #zdim*2

    def forward(self, x):
        #! ENCODER
        x = x.to(self.device)
        print(x.shape)
        x = x.to('cpu').numpy()
        x_real = x[:,:-1]
        x_real = torch.tensor(x_real)
        
        x_bin = x[:,-1]
        x_bin = torch.tensor(x_bin)
        
        qs = []
        ps = []
        
        qs_real, qs_bin , zs_real, zs_bin = self.forward_diff(x_real, x_bin, self.dif_depth, self.beta)
            
        # for t in torch.stack(zs)[:, :, 0].T[:100]:
        #     t = t.cpu().numpy()
        #     plt.plot(t, c='navy', alpha=0.1)
        # plt.xlabel('Diffusion time')
        # plt.ylabel('Data')
        # plt.show()
        
        
        for idx in range(len(self.encoders) - 1, -1, -1):
            #print(idx)
            mu, log_sigma = self.encoders[idx](zs_real[idx+1])
            ps.append(torch.distributions.Normal(mu,torch.exp(log_sigma)))
        
        #! DECODER
        x_mu, x_sigma, x_p = self.decoder(zs[0])
        xhat_gauss = torch.cat((x_mu, x_sigma), dim=1)
        xhat = torch.cat((xhat_gauss, x_p.view(-1,1)), dim=1)
        pxz = Normal(x_mu, torch.exp(x_sigma))
        

        
        REC = -pxz.log_prob(x_real.to(self.device)).sum(dim=1)
        BCE_B = F.binary_cross_entropy(x_bin.to(self.device), x_p, reduction='sum')
        
        pz = torch.distributions.Normal(torch.zeros_like(x_real), torch.ones_like(x_real))
        
        print(pz)
        print(qs[-1])
        exit()
        KLD_LAST =  torch.distributions.kl_divergence(qs[-1], pz).sum(dim=1)
        KLD_FIRST = torch.distributions.kl_divergence(qs[0], ps[0]).sum(dim=1)
        
        KLD_REST = 0
        
        for idx in range(2, len(qs)):
            KLD_REST += torch.distributions.kl_divergence(qs[0], ps[0]).sum(dim=1)
        
        KLD_ALL = KLD_FIRST + KLD_REST + KLD_LAST
        
        
        # beta = 1.0

        # return torch.mean(REC_G + beta*(KLD_G_1 + KLD_G_2)) + 0.001*BCE_B 
        return torch.mean(REC + KLD_ALL)    
        


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

    def forward_diff(self, x_real, x_bin, depth, beta):
        distrs_real, distrs_bin, samples_real, samples_bin = [None], [None], [x_real], [x_bin]
        xt_real = x_real
        xt_bin = x_bin
        
        for t in range(depth):
            q_real = torch.distributions.Normal(np.sqrt(1 - beta)*xt_real, beta*torch.ones_like(xt_real))
            q_bin = torch.distributions.Bernoulli(np.sqrt(1 - beta)*xt_bin + 0.5*xt_bin*torch.ones_like(xt_bin))
            xt_real = q_real.sample()
            xt_bin = q_bin.sample()
            distrs_real.append(q_real)
            distrs_bin.append(q_bin)
            samples_real.append(xt_real)
            samples_bin.append(xt_bin)
            
        return distrs_real, distrs_bin, samples_real, samples_bin
            
