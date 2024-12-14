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
        

    def sample(self, mu, std, delta_mu, delta_std):
        qz_gauss = torch.distributions.Normal(mu + delta_mu, std * delta_std)
        # ones = torch.ones(delta_std.shape).to(self.device)
        # sigma_new_2 = (std * delta_std.pow(2))/(ones + delta_std.pow(2))
        # mu_new_2 = (mu * delta_std.pow(2) + delta_mu.pow(2) * std)/(ones + delta_std.pow(2))
        
        # qz_gauss = torch.distributions.Normal(mu_new_2, sigma_new_2)
        z = qz_gauss.rsample()
        return z, qz_gauss
    
    def log_probas(self,z, mu, std):
        qzx = Normal(mu,  std)
        E_log_pxz = qzx.log_prob(z).sum(dim=1)
        return E_log_pxz, qzx
        
    def forward(self, x):
        x_new = self.body1(x)
        res = x_new
        x_new = self.body2(x_new)
        #x_new = self.body3(x_new)
        #x_new = self.body4(x_new)
        #x_new += res
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
        self.conf_general_decode = config["generate"]["decoder"]
        
        layer_num = self.conf_general_decode["layer_num"]
        arch = self.conf_general_decode["architecture"]
        bNorm = self.conf_general_decode["batchNorm"]
        relu = self.conf_general_decode["relu"]
        drop = self.conf_general_decode["dropout"]
        
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
        
    def log_probas(self,x, mu, std):
        pxz = Normal(mu,  std)
        E_log_pxz = pxz.log_prob(x).sum(dim=1)
        return E_log_pxz, pxz

    def forward(self, z, feature_type_dict):
        z_new = self.body1(z)
        res = z_new
        #z_new += x_skip
        z_new = self.body2(z_new)
        #z_new = self.body3(z_new)
        #z_new = self.body4(z_new)
        #z_new += res
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


class Deterministic_Encoder(nn.Module):
    def __init__(self, zdim, input_size, config:dict):
        super(Deterministic_Encoder, self).__init__()
        self.zdim = zdim
        self.input_size = input_size
        self.conf_general_encode = config["generate"]["encoder_deterministic"]
        
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
        self.body5 = nn.Sequential(*res_block(arch[4][0], self.zdim, 4))
        
        
    def forward(self, x):
        x_new = self.body1(x)
        res = x_new
        x_new = self.body2(x_new)
        #x_new = self.body3(x_new)
        #x_new = self.body4(x_new)
        #x_new += res
        scores = self.body5(x_new)
        
        return scores

class VAE(nn.Module):
    def __init__(self, prior, zdim, device, input_size, config:dict, output_dim):
        super(VAE, self).__init__()
        self.device = device
        self.zdim = zdim
        self.input_size = input_size
        self.config = config
        self.r_dim = config['generate']['general']['r_size']
        self.decored_output_dim = output_dim

        
        self.deterministic_encoder_1 = Deterministic_Encoder(self.r_dim, self.input_size, self.config).to(self.device)
        self.deterministic_encoder_2 = Deterministic_Encoder(self.r_dim, self.r_dim, self.config).to(self.device)    
        self.encoder_1 = Encoder(self.zdim, self.r_dim, self.config).to(self.device) #zdim*2
        self.encoder_2 = Encoder(self.zdim, self.r_dim, self.config).to(self.device)
        self.encoder_3 = Encoder(self.zdim, self.zdim, self.config).to(self.device) #zdim*2
        self.decoder = Decoder(self.zdim, self.input_size, self.config, self.decored_output_dim).to(self.device) #zdim*2
        #! self.prior = prior Not used for LVAE

    def forward(self, x, feature_type_dict):
        #! ENCODER
        x = x.to(self.device)
        #!DETERMINISTIC PATH
        r_1 = self.deterministic_encoder_1(x)
        r_2 = self.deterministic_encoder_2(r_1)
        delta_mu_1, delta_std_1 = self.encoder_1(r_1.view(-1, self.r_dim))
        #delta_sigma_1 = F.hardtanh(delta_sigma_1, -7., 2.)
        delta_mu_2, delta_std_2 = self.encoder_2(r_2.view(-1, self.r_dim))
        #delta_sigma_2 = F.hardtanh(delta_sigma_2, -7., 2.)
        z_2, qz_gauss_2 = self.encoder_2.sample(0,1,delta_mu_2,delta_std_2)
        
        
        mu_1, std_1 = self.encoder_3(z_2.view(-1, self.zdim))
        
        z_1, qz_gauss_1 = self.encoder_1.sample(mu_1,std_1,delta_mu_1,delta_std_1)
        
        pz_2_gauss = torch.distributions.Normal(torch.zeros_like(delta_mu_2), torch.ones_like(delta_std_2))
        pz_1_gauss = torch.distributions.Normal(mu_1, std_1)
        
        #! DECODER
        mu_gauss, std_gauss, p_bernoulli, p_categorical = self.decoder(z_1, feature_type_dict)
        x_gauss = x[:,feature_type_dict['real_data']]
        x_bernoulli = (torch.sign(x[:,feature_type_dict['binary_data']])+1)/2
        #! LOSS
        
        BC = F.binary_cross_entropy(p_bernoulli, x_bernoulli, reduction='sum')
        RE, _ = self.decoder.log_probas(x_gauss,mu_gauss, std_gauss)  
        #KL = torch.distributions.kl_divergence(qz, pz_gauss).sum(dim=1)
        KLD_G_2 = torch.distributions.kl_divergence(qz_gauss_2, pz_2_gauss).sum(dim=1)
        KLD_G_1 = torch.distributions.kl_divergence(qz_gauss_1, pz_1_gauss).sum(dim=1)

        
        MC = sum(F.cross_entropy(p_categorical[:,start_p:end_p], x[:,start_x:end_x].argmax(dim=1), reduction='sum') for (start_x, end_x), (start_p, end_p) in zip(feature_type_dict['categorical_one_hot'], feature_type_dict['categorical_only']))

        beta = 0.01
        gamma = 0.01
        
        return -torch.mean(RE) + beta*BC + gamma*MC + torch.mean(KLD_G_1 + KLD_G_2)

        
        # x_gauss = x[:, :-1]
        # x_bernoulli = x[:, -1].to(torch.float32).to(self.device)
        
        # x_hat_gauss = x_hat[:, :-1]
        # x_hat_bernoulli = x_hat[:, -1].to(torch.float32).to(self.device)
        
        # x_bernoulli = torch.sigmoid(x_bernoulli)
        
        # REC_G = self.recon(x_hat_gauss, x_gauss)
        # KLD_G_1 = torch.distributions.kl_divergence(qz_gauss_1, pz_1_gauss).sum(dim=1)
        # KLD_G_2 = torch.distributions.kl_divergence(qz_gauss_2, pz_2_gauss).sum(dim=1)
        # BCE_B = F.binary_cross_entropy(x_bernoulli, x_hat_bernoulli, reduction='sum')
        
        # beta = 1.0

        # return torch.mean(REC_G + beta*(KLD_G_1 + KLD_G_2)) + 0.001*BCE_B 
        # #return torch.mean(REC_G + beta*(BCE_B + KLD_G))


    def count_params(self):
        return sum(p.numel() for p in self.encoder_1.parameters() if p.requires_grad)

    def kl_div(self,z,mu,std, dist_1, dist_2, type):
        kl_part_1 = None
        kl_part_2 = None
        # prior and posterior
        if type == 'prpo':
            kl_part_1 = dist_1.log_probas(z[1])
            #print(kl_part_1)
        # two posteriors
        elif type == 'popo':
            kl_part_1 = dist_1.log_probas(z[0], mu[0], std[0])[0]
        kl_part_2 = dist_2.log_probas(z[1], mu[1], std[1])[0] 
        #print(kl_part_2)
        #exit()
        KL = kl_part_1 - kl_part_2
        return KL


