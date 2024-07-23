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
    
    def sample(self, mu, std):
        qz_gauss = torch.distributions.Normal(mu, std)
        z = qz_gauss.sample()
        return z, qz_gauss
    
    def neg_log_prob(self,z, mu, std):
        qzx = Normal(mu,  std)
        logp_qzx = -qzx.log_prob(z).sum(dim=1)
        return logp_qzx, qzx
    
    def forward(self, x):
        scores = self.body(x)
        mu, sigma = torch.split(scores, self.zdim, dim=1)
        std = torch.exp(sigma)
        return mu, std 

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
            
            if bNorm[idx] != 0: layers.append(nn.BatchNorm1d(num_features=bNorm[idx]))
            #if bNorm[idx] != 0:layers.append(nn.LayerNorm(normalized_shape=bNorm[idx]))
            #if bNorm[idx] != 0:layers.append(nn.InstanceNorm1d(num_features=bNorm[idx]))
            if relu[idx] != 0: layers.append(nn.ReLU())
            #if drop[idx] != 0: layers.append(nn.Dropout(drop[idx]))
        
        self.body = nn.Sequential(*layers)


    def sample(self, mu, std):
        px_gauss = torch.distributions.Normal(mu, std)
        x = px_gauss.sample()
        return x, px_gauss
    
    def neg_log_prob(self,x, mu, std):
        pxz = Normal(mu,  std)
        logp_pxz = -pxz.log_prob(x).sum(dim=1)
        return logp_pxz, pxz
    
    def forward(self, z):
        xhat = self.body(z)
        xhat_gauss = xhat[:, :-1]
        xhat_bernoulli = xhat[:,-1]
        xhat_gauss_mu, xhat_gauss_sigma = torch.split(xhat_gauss, self.input_size - 1, dim=1)
        xhat_gauss_std = torch.exp(xhat_gauss_sigma)
        xhat_bernoulli = torch.sigmoid(xhat_bernoulli)
        return xhat_gauss_mu, xhat_gauss_std, xhat_bernoulli
    
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
        sample = sample.to(self.device)
        if step == 1:         
            mu, std = self.encoder(sample)
            z, qzx = self.encoder.sample(mu, std)  
            z = z.detach().clone()      
            mu, std, p = self.decoder(z)
            
            x_gauss = sample[:, :-1]
            x_bernoulli = sample[:, -1].to(torch.float32)
            logp_pxz, pxz = self.decoder.neg_log_prob(x_gauss, mu, std)
            x_bernoulli = torch.sigmoid(x_bernoulli)   
            
            LOSS_G = logp_pxz
            LOSS_B = F.binary_cross_entropy_with_logits(p, x_bernoulli, reduction="sum")
            return torch.mean(LOSS_G ) + 0.0000000000001*LOSS_B
            #return torch.mean(LOSS_G) 
            
        elif step == 2:
            mu_gauss, std_gauss, p_bernoulli = self.decoder(sample)
            x_gauss, pxz = self.decoder.sample(mu_gauss, std_gauss)
            x_bernoulli = torch.bernoulli(p_bernoulli)
            x = torch.cat((x_gauss, x_bernoulli.view(-1,1)), dim=1)
            mu, std = self.encoder(x.view(-1, self.input_size))

            logp_qzx, qzx = self.encoder.neg_log_prob(sample, mu, std)
            LOSS = logp_qzx 
            
            return torch.mean(LOSS) 

    def count_params(self):
        return sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)

        
