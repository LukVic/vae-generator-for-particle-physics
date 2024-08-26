import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import torch.nn.init as init
import optuna
import numpy as np
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
        
        # layers = []
        # for idx in range(layer_num):
        #     if idx == 0: 
        #         layers.append(nn.Linear(self.input_size, arch[idx][0]))
        #         #init.xavier_uniform_(layers[-1].weight)
        #     elif idx == layer_num - 1: 
        #         layers.append(nn.Linear(arch[idx][0], self.zdim*2))
        #         #init.xavier_uniform_(layers[-1].weight)
        #     else: 
        #         layers.append(nn.Linear(arch[idx][0],arch[idx][1]))
        #         #init.xavier_uniform_(layers[-1].weight)
            
        #     #if bNorm[idx] != 0: layers.append(nn.BatchNorm1d(num_features=bNorm[idx]))
        #     if bNorm[idx] != 0:layers.append(nn.LayerNorm(normalized_shape=bNorm[idx]))
        #     #if bNorm[idx] != 0:layers.append(nn.InstanceNorm1d(num_features=bNorm[idx]))
        #     #if relu[idx] != 0: layers.append(nn.ReLU())
        #     if relu[idx] != 0: layers.append(nn.GELU())
        #     if drop[idx] != 0: layers.append(nn.Dropout(drop[idx]))
        
        #self.body = nn.Sequential(*layers)

    def sample(self, mu, std):
        qz_gauss = torch.distributions.Normal(mu, std)
        z = qz_gauss.rsample()
        return z, qz_gauss
    
    def neg_log_prob(self,z, mu, std):
        qzx = Normal(mu,  std)
        E_log_pxz = -qzx.log_prob(z).sum(dim=1)
        return E_log_pxz, qzx
        
    def forward(self, x):
        x_new = self.body1(x)
        res = x_new
        x_new = self.body2(x_new)
        x_new = self.body3(x_new)
        x_new = self.body4(x_new)
        x_new += res
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
        self.conf_general_encode = config["generate"]["encoder"]
        
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
        
        
        
        # for idx in range(layer_num):
        #     if idx == 0: 
        #         layers.append(nn.Linear(self.zdim, arch[idx][0]))
        #         #init.xavier_uniform_(layers[-1].weight)
        #     elif idx == layer_num - 1: 
        #         layers.append(nn.Linear(arch[idx][0], self.input_size*2-1))
        #         #init.xavier_uniform_(layers[-1].weight)
        #     else: 
        #         layers.append(nn.Linear(arch[idx][0],arch[idx][1]))
        #         #init.xavier_uniform_(layers[-1].weight)
            
        #     #if bNorm[idx] != 0: layers.append(nn.BatchNorm1d(num_features=bNorm[idx]))
        #     if bNorm[idx] != 0:layers.append(nn.LayerNorm(normalized_shape=bNorm[idx]))
        #     #if bNorm[idx] != 0:layers.append(nn.InstanceNorm1d(num_features=bNorm[idx]))
        #     #if relu[idx] != 0: layers.append(nn.ReLU())
        #     if relu[idx] != 0: layers.append(nn.GELU())
        #     #if relu[idx] != 0: layers.append(nn.Sigmoid())
        #     if drop[idx] != 0: layers.append(nn.Dropout(drop[idx]))
        
        #self.body = nn.Sequential(*layers)
    
    def neg_log_prob(self,x, mu, std):
        pxz = Normal(mu,  std)
        E_log_pxz = -pxz.log_prob(x).sum(dim=1)
        return E_log_pxz, pxz

    def forward(self, z, feature_type_dict):
        print(feature_type_dict)
        
        z_new = self.body1(z)
        res = z_new
        #z_new += x_skip
        z_new = self.body2(z_new)
        z_new = self.body3(z_new)
        z_new = self.body4(z_new)
        z_new += res
        xhat = self.body5(z_new)
        
        # Gaussian
        xhat_gauss = xhat[:, :-1]
        xhat_gauss_mu, xhat_gauss_sigma = torch.split(xhat_gauss, self.input_size - 1, dim=1)
        xhat_gauss_std = torch.exp(xhat_gauss_sigma)
        
        # Bernoulli
        xhat_bernoulli = []
        for val in feature_type_dict['binary']:
            xhat_bernoulli.append(torch.sigmoid(xhat[:,val[0]]))
        return xhat_gauss_mu, xhat_gauss_std, xhat_bernoulli




class VAE(nn.Module):
    def __init__(self, zdim, device, input_size, config:dict, output_dim):
        super(VAE, self).__init__()
        self.device = device
        self.zdim = zdim
        self.input_size = input_size
        self.config = config
        self.decoder_output_dim = output_dim
        
        self.encoder = Encoder(self.zdim, self.input_size, self.config).to(self.device)
        self.decoder = Decoder(self.zdim, self.input_size, self.config, self.decoder_output_dim).to(self.device)
        #self.prior = VampPrior(self.zdim, self.input_size, 255, self.encoder,16,None)
        self.prior = MoGPrior(self.zdim, self.zdim)
        #self.prior = StandardPrior(self.zdim)
        
    def forward(self, x, feature_type_dict):
        #! ENCODER
        x = x.to(self.device)
        mu, std = self.encoder(x.view(-1, self.input_size))
        z, qz = self.encoder.sample(mu, std)

        pz_gauss = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        #! DECODER
        # TODO
        mu_gauss, std_gauss, p_bernoulli = self.decoder(z, feature_type_dict)
        
        x_gauss = x[:, :-1].to(self.device)
        x_bernoulli = x[:, -1].to(torch.float32).to(self.device)
        x_bernoulli = torch.sigmoid(x_bernoulli)
        
        RE, _ = self.decoder.neg_log_prob(x_gauss,mu_gauss, std_gauss)  
        KL = torch.distributions.kl_divergence(qz, pz_gauss).sum(dim=1)
        #KL = - self.prior.log_prob(z) + self.encoder.log_prob(z, mu, std)[0]
        
        BC = F.binary_cross_entropy(x_bernoulli, p_bernoulli, reduction='sum')
        #BC = F.binary_cross_entropy_with_logits(x_bernoulli,p_bernoulli, reduction="sum")
        beta = 0.01

        #return torch.mean(RE + beta*KL) 
        return torch.mean(RE + beta*BC + KL)

    def count_params(self):
        return sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)

    
    
    
class StandardPrior(nn.Module):
    def __init__(self, L=2):
        super(StandardPrior, self).__init__()

        self.L = L 

        # params weights
        self.means = torch.zeros(1, L)
        self.logvars = torch.zeros(1, L)

    def get_params(self):
        return self.means, self.logvars

    def sample(self, batch_size):
        return torch.randn(batch_size, self.L)
    
    def log_prob(self, z):
        return self.log_standard_normal(z)

    def log_standard_normal(self, x, reduction=None, dim=None):
        PI = torch.from_numpy(np.asarray(np.pi))
        log_p = -0.5 * torch.log(2. * PI) - 0.5 * x**2.
        if reduction == 'avg':
            return torch.mean(log_p, dim)
        elif reduction == 'sum':
            return torch.sum(log_p, dim)
        else:
            return log_p.T

class MoGPrior(nn.Module):
    def __init__(self, L, num_components):
        super(MoGPrior, self).__init__()

        self.L = L
        self.num_components = num_components

        multiplier = 1

        self.means = nn.Parameter(torch.randn(num_components, self.L)*multiplier)
        self.logvars = nn.Parameter(torch.randn(num_components, self.L))

        self.w = nn.Parameter(torch.zeros(num_components, 1, 1))

    def get_params(self):
        return self.means, self.logvars

    def sample(self, batch_size):
        # mu, lof_var
        means, logvars = self.get_params()

        w = F.softmax(self.w, dim=0)
        w = w.squeeze()

        indexes = torch.multinomial(w, batch_size, replacement=True)

        eps = torch.randn(batch_size, self.L)
        for i in range(batch_size):
            indx = indexes[i]
            if i == 0:
                z = means[[indx]] + eps[[i]] * torch.exp(logvars[[indx]])
            else:
                z = torch.cat((z, means[[indx]] + eps[[i]] * torch.exp(logvars[[indx]])), 0)
        return z

    def log_prob(self, z):
        means, logvars = self.get_params()

        w = F.softmax(self.w, dim=0)

        z = z.unsqueeze(0) 
        means = means.unsqueeze(1) 
        logvars = logvars.unsqueeze(1) 

        log_p = self.log_normal_diag(z.to('cuda'), means.to('cuda'), logvars.to('cuda')) + torch.log(w).to('cuda') # K x B x L
        log_prob = torch.logsumexp(log_p, dim=0, keepdim=False) 
        return -log_prob.T

    def log_normal_diag(self, x, mu, log_var, reduction=None, dim=None):
        PI = torch.from_numpy(np.asarray(np.pi))
        log_p_1 = -0.5 * torch.log(2. * PI) - 0.5 * log_var
        log_p_2 = - 0.5 * torch.exp(-log_var) * (x - mu)**2.
        log_p = log_p_1 + log_p_2
        if reduction == 'avg':
            return torch.mean(log_p, dim)
        elif reduction == 'sum':
            return torch.sum(log_p, dim)
        else:
            return log_p