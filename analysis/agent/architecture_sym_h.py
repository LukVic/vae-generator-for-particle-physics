import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import torch.nn.init as init
import optuna
from torchviz import make_dot
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
            
            # if bNorm[idx] != 0: layers.append(nn.BatchNorm1d(num_features=bNorm[idx]))
            if bNorm[idx] != 0:layers.append(nn.LayerNorm(normalized_shape=bNorm[idx]))
            #if bNorm[idx] != 0:layers.append(nn.InstanceNorm1d(num_features=bNorm[idx]))
            if relu[idx] != 0: layers.append(nn.ReLU())
            #if relu[idx] != 0: layers.append(nn.GELU())
            #if drop[idx] != 0: layers.append(nn.Dropout(drop[idx]))
        
        self.body = nn.Sequential(*layers)
        

    def encode(self, x):
        scores = self.body(x)
        mu, sigma = torch.split(scores, self.zdim, dim=1)
        std = torch.exp(sigma)
        return mu, std  

    
    def log_prob(self,z, mu, std):
        qzx = Normal(mu,  std)
        logp_qzx = -qzx.log_prob(z).sum(dim=1)
        return logp_qzx, qzx
    
    def forward(self, x):
        pass

class Encoder_Linear(nn.Module):
    def __init__(self, zdim, input_size, config:dict):
        super(Encoder_Linear, self).__init__()
        self.zdim = zdim
        self.input_size = input_size
        self.config = config
        
        layer_num = config["encoder_linear"]["layer_num"]
        arch = config["encoder_linear"]["architecture"]
        bNorm = config["encoder_linear"]["batchNorm"]
        relu = config["encoder_linear"]["relu"]
        drop = config["encoder_linear"]["dropout"]
        
        
        layers = []
        layers.append(nn.Linear(self.zdim, self.zdim*2))
        layers.append(nn.GELU())
        layers.append(nn.BatchNorm1d(self.zdim*2))
        
        self.body = nn.Sequential(*layers)
        

    def encode(self, x):
        scores = self.body(x)
        mu, sigma = torch.split(scores, self.zdim, dim=1)
        std = torch.exp(sigma)
        return mu, std  

    
    def log_prob(self,z, mu, std):
        qzx = Normal(mu,  std)
        logp_qzx = -qzx.log_prob(z).sum(dim=1)
        return logp_qzx, qzx
    
    def forward(self, x):
        pass


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
            
            # if bNorm[idx] != 0: layers.append(nn.BatchNorm1d(num_features=bNorm[idx]))
            if bNorm[idx] != 0:layers.append(nn.LayerNorm(normalized_shape=bNorm[idx]))
            #if bNorm[idx] != 0:layers.append(nn.InstanceNorm1d(num_features=bNorm[idx]))
            if relu[idx] != 0: layers.append(nn.ReLU())
            #if relu[idx] != 0: layers.append(nn.GELU())
            #if relu[idx] != 0: layers.append(nn.Sigmoid())
            #if drop[idx] != 0: layers.append(nn.Dropout(drop[idx]))
        
        self.body = nn.Sequential(*layers)

    def decode(self, z):
        xhat = self.body(z)
        xhat_gauss = xhat[:, :-1]
        xhat_bernoulli = xhat[:,-1]
        xhat_gauss_mu, xhat_gauss_sigma = torch.split(xhat_gauss, self.input_size - 1, dim=1)
        xhat_gauss_std = torch.exp(xhat_gauss_sigma)
        xhat_bernoulli = torch.sigmoid(xhat_bernoulli)
        return xhat_gauss_mu, xhat_gauss_std, xhat_bernoulli
    
    def log_prob(self,x, mu, std):
        pxz = Normal(mu,  std)
        logp_pxz = -pxz.log_prob(x).sum(dim=1)
        return logp_pxz, pxz
    
    def forward(self, z):
        pass
        


class Deterministic_encoder(nn.Module):
    def __init__(self, input_size, r_size ,config:dict):
        super(Deterministic_encoder, self).__init__()
        self.r_size = r_size
        self.input_size = input_size
        self.config = config
        
        layer_num = config["encoder_deterministic"]["layer_num"]
        arch = config["encoder_deterministic"]["architecture"]
        bNorm = config["encoder_deterministic"]["batchNorm"]
        relu = config["encoder_deterministic"]["relu"]
        drop = config["encoder_deterministic"]["dropout"]
        
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
            
            # if bNorm[idx] != 0: layers.append(nn.BatchNorm1d(num_features=bNorm[idx]))
            if bNorm[idx] != 0:layers.append(nn.LayerNorm(normalized_shape=bNorm[idx]))
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

        print(self.input_size)
        #self.deterministic_encoder_1 = Deterministic_encoder(self.input_size, self.r_dim, self.config).to(self.device)
        #self.deterministic_encoder_2 = Deterministic_encoder(self.r_dim, self.r_dim, self.config).to(self.device)    
        # self.encoder_1 = Encoder(self.zdim, self.r_dim, self.config).to(self.device)
        # self.encoder_2 = Encoder(self.zdim, self.r_dim, self.config).to(self.device)
        # self.coder = Encoder(self.zdim, self.zdim, self.config).to(self.device)
        # self.decoder = Decoder(self.zdim, self.input_size, self.config).to(self.device)

        self.deterministic_encoders = nn.ModuleList([
            Deterministic_encoder(self.input_size, self.r_dim, self.config).to(self.device),
            Deterministic_encoder(self.r_dim, self.r_dim, self.config).to(self.device)
        ])
        
        self.encoders = nn.ModuleList([
            Encoder(self.zdim, self.r_dim, self.config).to(self.device),
            Encoder(self.zdim, self.r_dim, self.config).to(self.device)
        ])
        
        self.decoders = nn.ModuleList([
            Encoder(self.zdim, self.zdim, self.config).to(self.device),       
            Decoder(self.zdim, self.input_size, self.config).to(self.device)
        ])
        
    def forward(self, sample, step):
        #! THE FIRST STEP
        if step == 1:
            
            #! ENCODE AND SAMPLE
            x = sample.to(self.device)
            x_gauss = x[:, :-1]

            r_1 = self.deterministic_encoders[0](x)
            delta_mu_1, delta_std_1 = self.encoders[0].encode(r_1)
            
            # delta_mu_1 = delta_mu_1.detach()
            # delta_std_1 = delta_std_1.detach()
            
            r_2 = self.deterministic_encoders[1](r_1)
            delta_mu_2, delta_std_2 = self.encoders[1].encode(r_2)
            
            # delta_mu_2 = delta_mu_2.detach()
            # delta_std_2 = delta_std_2.detach()
            
            ones = torch.ones(delta_std_1.shape).to(self.device)
            
            mu_2_shifted, std_2_shifted = self.distr_shift(0,ones,delta_mu_2,delta_std_2,'easy')
            
            
            z_2, _ = self.sample(mu_2_shifted, std_2_shifted)
            z_2 = z_2.detach()
            
            mu_1, std_1 = self.decoders[0].encode(z_2)
            
            mu_1_shifted, std_1_shifted = self.distr_shift(mu_1, std_1, delta_mu_1, delta_std_1, 'easy')
            z_1, _ = self.sample(mu_1_shifted, std_1_shifted)
            z_1 = z_1.detach()
            

            mu_1 = mu_1.detach().clone()
            std_1 = std_1.detach().clone()
            
            #! LEARN
            mu_z, std_z = self.decoders[0].encode(z_2)
            mu_x, std_x, p_x = self.decoders[1].decode(z_1)
            
            
            #mu_z = mu_z.detach().clone()
            #std_z = std_z.detach().clone()

            
            logp_pxz_1, _ = self.decoders[1].log_prob(x_gauss, mu_x, std_x)
            logp_pz_1z_2, _ = self.decoders[0].log_prob(z_1, mu_z, std_z)
            
            dot = make_dot(std_1, params=dict(self.decoders[0].named_parameters()))
            dot.render("computational_graph", format="png")

            D_LOSS = torch.mean(logp_pxz_1)
            #D_LOSS = torch.mean(logp_pz_1z_2)
            #D_LOSS = torch.mean(logp_pxz_1 + logp_pz_1z_2)

            return D_LOSS
        
        #! THE SECOND STEP
        elif step == 2:
            z_2 = sample.to(self.device)
                        
            mu_1, std_1 = self.decoders[0].encode(z_2.view(-1, self.zdim))
            z_1, _ = self.sample(mu_1, std_1)
            #z_1 = z_1.detach()
            
            
            mu_gauss, std_gauss, p_bernoulli = self.decoders[1].decode(z_1)
            #print(std_gauss)
            x_gauss, _ = self.sample(mu_gauss, std_gauss)
            x_gauss = x_gauss.detach()
            x_bernoulli = torch.bernoulli(p_bernoulli)
            x = torch.cat((x_gauss, x_bernoulli.view(-1,1)), dim=1)
            
            
            #! LEARN
            r_1 = self.deterministic_encoders[0](x)
            r_2 = self.deterministic_encoders[1](r_1)
            delta_mu_1, delta_std_1 = self.encoders[0].encode(r_1)
            delta_mu_2, delta_std_2 = self.encoders[1].encode(r_2)
            
            ones = torch.ones(delta_std_1.shape).to(self.device)
            
            
            mu_2_shifted, std_2_shifted = self.distr_shift(0, ones, delta_mu_2, delta_std_2,'easy')
            logp_pz_2x, _ = self.encoders[1].log_prob(z_2, mu_2_shifted, std_2_shifted)
            mu_z_1, std_z_1 = self.decoders[0].encode(z_2)
            
            mu_z_1 = mu_z_1.detach()
            std_z_1 = std_z_1.detach()
            
            mu_1_shifted, std_1_shifted = self.distr_shift(mu_z_1, std_z_1, delta_mu_1, delta_std_1,'easy')
            logp_pz_1z_2x, _ = self.encoders[0].log_prob(z_1, mu_1_shifted, std_1_shifted)
            
            E_LOSS = torch.mean(logp_pz_1z_2x)
            #E_LOSS = torch.mean(logp_pz_2x)
            #E_LOSS = torch.mean(logp_pz_2x +logp_pz_1z_2x)
            
            return E_LOSS 

    def count_params(self):
        return sum(p.numel() for p in self.encoder_1.parameters() if p.requires_grad)

    def sample(self, mu, std):
        px_gauss = torch.distributions.Normal(mu, std)
        x = px_gauss.sample()
        return x, px_gauss
    
    def distr_shift(self, mu, std, delta_mu, delta_std, mode):
        if mode == 'easy':
            std_shifted = std * delta_std
            mu_shifted = mu + delta_mu
            
        elif mode == 'advn':
            std_shifted = std.pow(2) * delta_std.pow(2) / (std.pow(2) + delta_std.pow(2))
            mu_shifted = (std.pow(2) * delta_mu + mu * delta_std.pow(2)) / (std.pow(2) + delta_std.pow(2))
            
        return mu_shifted, std_shifted