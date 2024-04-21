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
            #if relu[idx] != 0: layers.append(nn.GELU())
            #if drop[idx] != 0: layers.append(nn.Dropout(drop[idx]))
        
        self.body = nn.Sequential(*layers)
        

    def encode(self, x):
        scores = self.body(x)
        mu, sigma = torch.split(scores, self.zdim, dim=1)
        std = torch.exp(sigma)
        return mu, std  

    def sample(self, mu, std):
        qz_gauss = torch.distributions.Normal(mu, std)
        z = qz_gauss.sample()
        return z, qz_gauss
    
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
            
            if bNorm[idx] != 0: layers.append(nn.BatchNorm1d(num_features=bNorm[idx]))
            #if bNorm[idx] != 0:layers.append(nn.LayerNorm(normalized_shape=bNorm[idx]))
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
    
    def sample(self, mu, std):
        px_gauss = torch.distributions.Normal(mu, std)
        x = px_gauss.sample()
        return x, px_gauss
    
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
        
        layer_num = config["decoder"]["layer_num"]
        arch = config["decoder"]["architecture"]
        bNorm = config["decoder"]["batchNorm"]
        relu = config["decoder"]["relu"]
        drop = config["decoder"]["dropout"]
        
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
            
            if bNorm[idx] != 0: layers.append(nn.BatchNorm1d(num_features=bNorm[idx]))
            #if bNorm[idx] != 0:layers.append(nn.LayerNorm(normalized_shape=bNorm[idx]))
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

         
        #self.deterministic_encoder_1 = Deterministic_encoder(self.input_size, self.r_dim, self.config).to(self.device)
        #self.deterministic_encoder_2 = Deterministic_encoder(self.r_dim, self.r_dim, self.config).to(self.device)    
        # self.encoder_1 = Encoder(self.zdim, self.r_dim, self.config).to(self.device)
        # self.encoder_2 = Encoder(self.zdim, self.r_dim, self.config).to(self.device)
        # self.coder = Encoder(self.zdim, self.zdim, self.config).to(self.device)
        # self.decoder = Decoder(self.zdim, self.input_size, self.config).to(self.device)

        self.deterministic_encoders = nn.ModuleList([
            Deterministic_encoder(self.input_size, self.r_dim, self.config).to(self.device),
            Deterministic_encoder(self.r_dim, self.r_dim, self.config).to(self.device),
            Deterministic_encoder(self.r_dim, self.r_dim, self.config).to(self.device)
        ])
        
        self.encoders = nn.ModuleList([
            Encoder(self.zdim, self.r_dim, self.config).to(self.device),
            Encoder(self.zdim, self.r_dim, self.config).to(self.device),
            Encoder(self.zdim, self.r_dim, self.config).to(self.device)
        ])
        
        self.decoders = nn.ModuleList([
            Encoder(self.zdim, self.zdim, self.config).to(self.device),
            Encoder(self.zdim, self.zdim, self.config).to(self.device),       
            Decoder(self.zdim, self.input_size, self.config).to(self.device)
        ])
        
    def forward(self, sample, step):
        num_layers = len(self.deterministic_encoders)
        
        #! THE FIRST STEP
        if step == 1:
            
            #! ENCODE AND SAMPLE
            x = sample.to(self.device)
            x_gauss = x[:, :-1]
            
            r = x 
            distr_shifts = []
            
            for idx in range(num_layers):
                r = self.deterministic_encoders[idx](r)
                delta_mu, delta_std = self.encoders[0].encode(r)
                distr_shifts.append([delta_mu, delta_std])
            
            mu = 0
            std = torch.ones(distr_shifts[0][0].shape).to(self.device)
            
            zs = []
            distr_params = []
            for idx in range(num_layers):
                delta_mu = distr_shifts[num_layers-idx-1][0]
                delta_std = distr_shifts[num_layers-idx-1][1]
                
                mu_shifted, std_shifted = self.distr_shift(mu,std,delta_mu,delta_std,'advn')
                z, _ = self.encoders[num_layers-idx-1].sample(mu_shifted, std_shifted)
                
                if idx != num_layers-1:
                    mu, std = self.decoders[idx].encode(z)
                else:
                    mu, std, p = self.decoders[idx].decode(z) 
                
                zs.append(z)
                distr_params.append([mu,std])
            
            #! LEARN
            logp_pz_2z_3, _ = self.decoders[0].log_prob(zs[1], distr_params[0][0], distr_params[0][1])
            logp_pz_1z_2, _ = self.decoders[1].log_prob(zs[2], distr_params[1][0], distr_params[1][1])
            logp_pxz_1, _ = self.decoders[2].log_prob(x_gauss, distr_params[2][0], distr_params[2][1])

            D_LOSS = torch.mean(logp_pxz_1 + logp_pz_1z_2 + logp_pz_2z_3)

            return D_LOSS
        
        #! THE SECOND STEP
        elif step == 2:
            z_3 = sample.to(self.device)
            
            zs = [z_3]
            distr_params = []
            
            for idx in range(num_layers):
                if idx != num_layers-1:
                    mu, std = self.decoders[idx].encode(zs[idx])
                    z, _ = self.decoders[idx].sample(mu, std)
                zs.append(z)
                distr_params.append([mu,std])
            
            
            mu_gauss, std_gauss, p_bernoulli = self.decoders[2].decode(zs[-1])
            x_gauss, _ = self.decoders[2].sample(mu_gauss, std_gauss)
            x_bernoulli = torch.bernoulli(p_bernoulli)
            x = torch.cat((x_gauss, x_bernoulli.view(-1,1)), dim=1)
            
            #! LEARN
            r_1 = self.deterministic_encoders[0](x)
            r_2 = self.deterministic_encoders[1](r_1)
            r_3 = self.deterministic_encoders[2](r_2)
            delta_mu_1, delta_std_1 = self.encoders[0].encode(r_1)
            delta_mu_2, delta_std_2 = self.encoders[1].encode(r_2)
            delta_mu_3, delta_std_3 = self.encoders[2].encode(r_3)
            
            ones = torch.ones(delta_std_3.shape).to(self.device)
            
            mu_3_shifted, std_3_shifted = self.distr_shift(0, ones, delta_mu_3, delta_std_3,'advn')
            logp_pz_3x, _ = self.encoders[2].log_prob(zs[0], mu_3_shifted, std_3_shifted)
            mu_z_2, std_z_2 = self.decoders[0].encode(zs[0])
            
            mu_z_2 = mu_z_2.detach().clone()
            std_z_2 = std_z_2.detach().clone()
            
            mu_2_shifted, std_2_shifted = self.distr_shift(mu_z_2, std_z_2, delta_mu_2, delta_std_2,'advn')
            logp_pz_2z_3x, _ = self.encoders[1].log_prob(zs[1], mu_2_shifted, std_2_shifted)
            mu_z_1, std_z_1 = self.decoders[1].encode(zs[1])
            
            mu_z_1 = mu_z_1.detach().clone()
            std_z_1 = std_z_1.detach().clone()
            
            mu_1_shifted, std_1_shifted = self.distr_shift(mu_z_1, std_z_1, delta_mu_1, delta_std_1,'advn')
            logp_pz_1z_2x, _ = self.encoders[0].log_prob(zs[2], mu_1_shifted, std_1_shifted)
            

            #E_LOSS = torch.mean(logp_pz_1z_2x)
            E_LOSS = torch.mean(logp_pz_3x+ logp_pz_2z_3x +logp_pz_1z_2x)
            
            return E_LOSS 

    def count_params(self):
        return sum(p.numel() for p in self.encoder_1.parameters() if p.requires_grad)

    
    def distr_shift(self, mu, std, delta_mu, delta_std, mode):
        if mode == 'easy':
            std_shifted = std * delta_std
            mu_shifted = mu + delta_mu
            
        elif mode == 'advn':
            std_shifted = std * delta_std / (std + delta_std)
            mu_shifted = (std * delta_mu + mu * delta_std) / (std + delta_std)
            
        return mu_shifted, std_shifted