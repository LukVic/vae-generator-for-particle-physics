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
        # x_new = self.body3(x_new)
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
        z_new = self.body1(z)
        res = z_new
        #z_new += x_skip
        z_new = self.body2(z_new)
        # z_new = self.body3(z_new)
        z_new = self.body4(z_new)
        z_new += res
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
        mu_gauss, std_gauss, p_bernoulli, p_categorical = self.decoder(z, feature_type_dict)
        #print(torch.sum(p_categorical[:,0:4],dim=1))
        x_gauss = x[:,feature_type_dict['real_data']]
        x_bernoulli = (torch.sign(x[:,feature_type_dict['binary_data']])+1)/2
        # print(x_bernoulli)
        # print(p_bernoulli)
        BC = F.binary_cross_entropy(p_bernoulli, x_bernoulli, reduction='sum')
        RE, _ = self.decoder.neg_log_prob(x_gauss,mu_gauss, std_gauss)  
        KL = torch.distributions.kl_divergence(qz, pz_gauss).sum(dim=1)
        KL = self.kl_div(z,mu,std, self.prior, self.encoder)
        MC = sum(F.cross_entropy(p_categorical[:,start_p:end_p], x[:,start_x:end_x].argmax(dim=1), reduction='sum') for (start_x, end_x), (start_p, end_p) in zip(feature_type_dict['categorical_one_hot'], feature_type_dict['categorical_only']))

        beta = 0.01
        gamma = 0.01
        
        return torch.mean(RE) - torch.mean(KL) + beta*BC + gamma*MC


    def count_params(self):
        return sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)

    def kl_div(self,z,mu,std, prior, posterior):
        kl_part_1 = prior.log_prob(z).sum(0) 
        kl_part_2 = posterior.neg_log_prob(z, mu, std)[0] 

        KL = kl_part_1 + kl_part_2
        return KL
    
    
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
        multiplier = 1
        self.L = L
        self.num_components = num_components

        # params
        self.means = nn.Parameter(torch.randn(num_components, self.L)*multiplier)
        self.logvars = nn.Parameter(torch.randn(num_components, self.L))

        # mixing weights
        self.w = nn.Parameter(torch.zeros(num_components, 1, 1))

    def get_params(self):
        return self.means, self.logvars

    def sample(self, batch_size):
        # mu, lof_var
        means, logvars = self.get_params()

        # mixing probabilities
        w = F.softmax(self.w, dim=0)
        w = w.squeeze()

        # pick components
        indexes = torch.multinomial(w, batch_size, replacement=True)

        # means and logvars
        eps = torch.randn(batch_size, self.L)
        for i in range(batch_size):
            indx = indexes[i]
            if i == 0:
                z = means[[indx]] + eps[[i]] * torch.exp(logvars[[indx]])
            else:
                z = torch.cat((z, means[[indx]] + eps[[i]] * torch.exp(logvars[[indx]])), 0)
        return z

    def log_prob(self, z):
        # mu, lof_var
        means, logvars = self.get_params()

        # mixing probabilities
        w = F.softmax(self.w, dim=0)

        # log-mixture-of-Gaussians
        z = z.unsqueeze(0).to('cuda:0') # 1 x B x L
        means = means.unsqueeze(1).to('cuda:0') # K x 1 x L
        logvars = logvars.unsqueeze(1).to('cuda:0') # K x 1 x L

        log_p = self.log_normal_diag(z, means, logvars) + torch.log(w) # K x B x L
        log_prob = torch.logsumexp(log_p, dim=0, keepdim=False) # B x L

        return log_prob

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