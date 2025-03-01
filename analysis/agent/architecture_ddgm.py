import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import torch.nn.init as init
import numpy as np
import matplotlib.pyplot as plt
import math
from gm_helpers import infer_feature_type
import copy

class Decoder(nn.Module):
    def __init__(self, input_size, output_size, config:dict):
        super(Decoder, self).__init__()
        self.input_dim = input_size
        self.output_dim = output_size
        self.conf_general_decode = config["generate"]["backward_diffusion_decoder"]
        
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
        
        self.body1 = nn.Sequential(*res_block(self.input_dim+1, arch[0][0], 0)) # add + 1 for each binary feature
        self.body2 = nn.Sequential(*res_block(arch[1][0],arch[1][1], 1))
        self.body3 = nn.Sequential(*res_block(arch[2][0],arch[2][1], 2))
        self.body4 = nn.Sequential(*res_block(arch[3][0],arch[3][1], 3))
        self.body5 = nn.Sequential(*res_block(arch[4][0], self.output_dim+1, 4)) 
    
    def time_embedding(self, t, batch_size):
        t = torch.tensor([t], dtype=torch.float32)
        t = t * 1000  # Scale time step
        # Sinusoidal time embedding (from Nichol, 2021; Dhariwal & Nichol, 2021)
        emb = torch.cat([torch.sin(t), torch.cos(t)], dim=-1)
        emb = nn.Linear(emb.size(-1), batch_size)(emb)  # Project to 128-dimension
        
        return emb 

    def forward(self, z, t, feature_type_dict):
        
        # Time embedding
        t_emb = self.time_embedding(t, z.size(0)).to('cuda')
        t_emb = t_emb.unsqueeze(1)
        t_emb = t_emb.repeat(1, z.size(1))
        #print(t_emb.shape
        # Concatenate time embedding with input z
        z_new = torch.cat([z, t_emb], dim=-1)       
        
        z_new = self.body1(z)
        res = z_new
        #z_new += x_skip
        z_new = self.body2(z_new)
        #z_new = self.body3(z_new)
        #z_new = self.body4(z_new)
        #z_new += res
        xhat = self.body5(z_new)

        xhat_gauss_noise = xhat[:, feature_type_dict['real_data'][0]:feature_type_dict['real_data'][-1]+1]
        xhat_bin_noise = [xhat[:, val[0]:(val[1] + 1)] for val in feature_type_dict['binary_param']]
        xhat_cat_noise = [xhat[:, val[0]+1:val[1]+1] for val in feature_type_dict['categorical_one_hot']]

        xhat_cat_noise = xhat_bin_noise + xhat_cat_noise  # Concatenate lists
        
        return xhat_gauss_noise, xhat_cat_noise


class DDGM(nn.Module):
    def __init__(self, prior, zdim, device, input_size, config:dict, output_dim):
        super(DDGM, self).__init__()
        self.device = device

        self.zdim = zdim
        self.input_size = input_size
        self.config = config
        self.decoder_output_dim = output_dim
        
        self.hidden_dim = 512
        self.timesteps = 10
        # print(self.beta)
        # self.beta = torch.FloatTensor([0.9]).to(self.device)
        # print(self.beta)
        # exit()
        self.decoders = nn.ModuleList([
            Decoder(self.input_size, self.decoder_output_dim, self.config).to(self.device) 
            for _ in range(self.timesteps)
        ])
        self.prior = prior

        #Initialize weights
        def weights_init(m):
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
        for i in range(len(self.decoders)):
            self.decoders[i].apply(weights_init)

    def forward(self, x, feature_type_dict):
        #! DECODERS INSTEAD OF ENCODERS
        #! SEPARATELY FOR EACH FEATURE TYPE
        qs_gauss = []
        qs_cat = []
        ps = []
        x = x.to(self.device)
        #print(feature_type_dict)
        
        x_0_gauss = x[:, feature_type_dict['real_data']]  # Extract Gaussian features
        # Extract binary features and convert to one-hot
        x_bernoulli = [x[:, val] for val in feature_type_dict['binary_data']]
        x_bernoulli_one_hot = [F.one_hot((x_bin == 1).long(), num_classes=2).float() for x_bin in x_bernoulli]
        x_bernoulli_one_hot = torch.cat(x_bernoulli_one_hot, dim=1)  # Concatenate along feature dimension
        # Extract categorical features separately
        x_0s_cat = [x[:, val[0]:val[1]] for val in feature_type_dict['categorical_one_hot']]

        # Add binary one-hot features as the first categorical feature
        x_0s_cat.insert(0, x_bernoulli_one_hot)  

        cat_features_len = [val.shape[1] for val in x_0s_cat] # stores number of categories for each categorcial feature

        # Forward diffusion for Gaussian and separate categorical variables
        alpha_bars, alphas, betas = self.cosine_scheduler(self.timesteps)
        alpha_bars = torch.FloatTensor(alpha_bars).to(self.device)
        alphas = torch.FloatTensor(alphas).to(self.device)
        betas = torch.FloatTensor(betas).to(self.device)
        qs_gauss = []
        qs_cat = [[] for _ in x_0s_cat]  # Keep separate lists for each categorical variable

        eps = 1e-8  # Small value for numerical stability
        min_prob = 1e-6  # Ensures no exact zeros
        
        categories_num = [(val[1] - val[0]) + 1 if (val[1] - val[0]) == 1 else (val[1] - val[0])  
        for val in feature_type_dict['binary_param'] + feature_type_dict['categorical_param']]
        x_t_gauss = x_0_gauss.clone()
        x_ts_cat = copy.deepcopy(x_0s_cat)
        for t in range(1,self.timesteps + 1):
            beta_t = betas[t]  
            # Gaussian q(x_t|x_t-1)
            q_t_tm1_gauss = torch.distributions.Normal(loc=torch.sqrt(1 - beta_t) * x_t_gauss, scale=(torch.sqrt(beta_t)))
            x_t_gauss = q_t_tm1_gauss.rsample()
            #self.print_histo_step(x_t_gauss[:,0])
            qs_gauss.append(q_t_tm1_gauss)
            
            # Forward diffusion separately for each categorical variable
            for i, x_t_cat in enumerate(x_ts_cat):
                q_t_tm1_cat_probs = (1 - beta_t) * x_t_cat + beta_t / x_t_cat.shape[1]
                q_t_tm1_cat_probs = q_t_tm1_cat_probs / (q_t_tm1_cat_probs.sum(dim=1, keepdim=True) + eps)
                q_t_tm1_cat = torch.distributions.Categorical(probs=q_t_tm1_cat_probs)
                x_t_features = q_t_tm1_cat.sample()
                x_ts_cat[i] = F.one_hot(x_t_features, num_classes=categories_num[i])
                qs_cat[i].append(q_t_tm1_cat)
        
        # Sample from the Gaussian distribution at the last timestep
        x_T_gauss = qs_gauss[-1].rsample()
        # Sample from each categorical distribution and concatenate them
        x_T_categorical = torch.cat([q_cat[-1].sample().unsqueeze(1) for q_cat in qs_cat], dim=1)
        
        # prepare one-hot for the backward path
        x_T_one_hot_categorical_list = [
        F.one_hot(x_feat.long(), num_classes=categories)  # One-hot encode each feature
            for x_feat, categories in zip(x_T_categorical.T, categories_num)
        ]

        # Backward path
        ps_gauss = []
        ps_cat = [[] for _ in x_0s_cat]
        # Init the x_t variable
        x_t_gauss = x_T_gauss.clone()
        x_ts_cat = copy.deepcopy(x_T_one_hot_categorical_list)
        for t, decoder in zip(reversed(range(1, self.timesteps)),reversed(self.decoders)):  # Iterate over timesteps in reverse
            # Decode the noisy features
            x_t = torch.cat([x_t_gauss, torch.cat(x_ts_cat, dim=1)], dim=1)
            x_t_gauss_noise, x_0_cat_hat = decoder(x_t, t, feature_type_dict)

            # Retrieve precomputed values from the scheduler              
            alpha_bar_tm1 = alpha_bars[t-1]
            alpha_bar_t = alpha_bars[t] 
            alpha_t = alphas[t]
            beta_t = betas[t]
            
            # print(f"Current time step: {t}")
            # print(f"beta at t step: {beta_t}")
            # print(f"alpha at t step: {alpha_t}")
            # print(f"alpha bar at t-1 step: {alpha_bar_tm1}")
            # print(f"alpha bar at t step: {alpha_bar_t}")
            #print(x_t_gauss_noise)
            x_t_gauss_noise = torch.clamp(x_t_gauss_noise, min=-5, max=5)
            # Gaussian Backward Path (for numerical features)
            mu_theta = (1 / torch.sqrt(alpha_t)) * (x_t_gauss - ((beta_t) / torch.sqrt(1 - alpha_bar_t)) * x_t_gauss_noise)
            sigma_t = torch.clamp(torch.sqrt(beta_t * (1 - alpha_bar_tm1) / (1 - alpha_bar_t)), min=eps)
            
            # No noise at final step
            q_t_tm1_gauss = torch.distributions.Normal(loc=mu_theta, scale=sigma_t)
            x_t_gauss = q_t_tm1_gauss.rsample()

            ps_gauss.append(q_t_tm1_gauss)
            
            # Categorical Backward Path (for categorical features)
            for i, x_t_cat in enumerate(x_ts_cat):  # Iterate over each categorical feature
            
                # print(f"x_cat shape: {x_t_cat.shape}")
                # print(f"x_0_cat_hat[i] shape: {x_0_cat_hat[i].shape}")
                # print(f"x_noisy_one_hot_categorical shape: {x_noisy_one_hot_categorical.shape}")
                
                pi_cat = ((alpha_t * x_t_cat + (1 - alpha_t) / x_t_cat.shape[1]) *
                        (alpha_bar_tm1 * x_0_cat_hat[i] + (1 - alpha_bar_tm1) / x_0_cat_hat[i].shape[1]))
                
                # Normalize π to get valid probabilities
                pi_cat = pi_cat / (pi_cat.sum(dim=1, keepdim=True) + eps)
                pi_cat = torch.clamp(pi_cat, min=min_prob)
                # Sample from the categorical distribution
                q_t_tm1_cat = torch.distributions.Categorical(probs=pi_cat)
                x_t_features = q_t_tm1_cat.sample()
                x_t_cat = F.one_hot(x_t_features.long(), num_classes=(cat_features_len[i]))
                x_ts_cat[i] = x_t_cat
                ps_cat[i].append(q_t_tm1_cat)
            x_t = torch.cat([x_t_gauss, torch.cat(x_ts_cat, dim=1)], dim=1)
        
        x_reconstructed = torch.cat([x_t_gauss, torch.cat(x_ts_cat, dim=1)], dim=1)
        original_features = torch.cat([x_0_gauss, torch.cat(x_0s_cat, dim=1)], dim=1)
        # Calculate MSE between original features and reconstructed features
        

        mse = F.mse_loss(x_t_gauss[:,2], x_0_gauss[:,2])
        print(f'Mean Squared Error: {mse.item()}')
        #if mse < 600:
        self.print_histo_step(x_t_gauss[:,2])
        self.print_histo_step(x_0_gauss[:,2])
        
        # Compute ELBO
        # Prepare D_KL(q(xT|x0)||p(xT))
        # --------------------------------------------------------
        # Gaussian
        p_xT_gauss = torch.distributions.Normal(torch.zeros_like(x_0_gauss), torch.ones_like(x_0_gauss))

        q_xT_x0_gauss = torch.distributions.Normal(loc=torch.sqrt(alpha_bars[-1]) * x_0_gauss, scale=(torch.sqrt(1 - alpha_bars[-1])))
        # Categorical
        kl_divs_cat = [[] for _ in x_0s_cat] 
        for i, x_cat in enumerate(x_0s_cat):
                # Prepare p(xT) cat
                p_xT_cat = torch.full_like(x_cat, 1.0 / x_cat.shape[1])
                p_xT_cat = torch.distributions.Categorical(probs=p_xT_cat)
                
                # Prepare q(xT|x0)
                q_xT_x0_cat_probs = (1 - alpha_bars[-1]) * x_cat + alpha_bars[-1] / x_cat.shape[1]
                q_xT_x0_cat_probs = torch.clamp(q_xT_x0_cat_probs, min=min_prob)
                q_xT_x0_cat_probs = q_xT_x0_cat_probs / (q_xT_x0_cat_probs.sum(dim=1, keepdim=True) + eps)
                q_xT_x0_cat = torch.distributions.Categorical(probs=q_xT_x0_cat_probs)
                
                kl_divs_cat[i].append(q_xT_x0_cat)
        
        terminal_dkl_gauss = torch.distributions.kl.kl_divergence(q_xT_x0_gauss, p_xT_gauss).sum(dim=1).mean()
        
        # --------------------------------------------------------
        # Prepare -log_q(x1|x0)(p(x0|x1))
        # Gaussian
        q_x1_x0_samples_gauss = qs_gauss[0].rsample()
        p_x0_x1_gauss = ps_gauss[-1]
        nll_gaussian = -p_x0_x1_gauss.log_prob(q_x1_x0_samples_gauss).sum(dim=1).mean()
        
        # Categorical
        nll_cat = 0
        for i, (q_cat, p_cat) in enumerate(zip(qs_cat, ps_cat)):
            q_x1_x0_samples_cat = q_cat[0].sample()
            p_x0_x1_cat = p_cat[-1]
            nll_cat += -p_x0_x1_cat.log_prob(q_x1_x0_samples_cat)
        
        # Prepare E_q(x_t|x_0) D_KL(q(x_t-1|x_t,x_0)||p(x_t-1|x_t))
        # q(x_t-1|x_t,x_0)
        qs_true_gauss = []
        qs_true_cat = [[] for _ in x_0s_cat]
        for t in reversed(range(1,self.timesteps)):
            beta_t = betas[t]
            alpha_t = alphas[t]
            alpha_bar_tm1 = alpha_bars[t-1]
            alpha_bar_t = alpha_bars[t]
            # Gaussian
            q_xt_x0_gauss = torch.distributions.Normal(loc=torch.sqrt(alpha_bars[t]) * x_0_gauss, scale=(torch.sqrt(1 - alpha_bars[t])))
            x_t_gauss = q_xt_x0_gauss.rsample()
            mu_t = x_0_gauss*(torch.sqrt(alpha_bar_tm1)*beta_t)/(1 - alpha_bar_t) + x_t_gauss*(torch.sqrt(alpha_t)*(1-alpha_bar_tm1))/(1-alpha_bar_t)
            sigma_t = torch.clamp(torch.sqrt(beta_t * (1 - alpha_bar_tm1) / (1 - alpha_bar_t)), min=eps)
            q_x_tm1_x_t_x_0 = torch.distributions.Normal(loc=mu_t, scale=sigma_t)
            qs_true_gauss.append(q_x_tm1_x_t_x_0)
            # Categorical
            for i, x_0_cat in enumerate(x_0s_cat):
                q_x_t_x_0 = torch.distributions.Categorical(probs=alpha_bar_t*x_0_cat + (1-alpha_bar_t)/x_0_cat.shape[1])
                x_t_features = q_x_t_x_0.sample()
                x_t_cat = F.one_hot(x_t_features.long(), num_classes=(cat_features_len[i]))
                pi_cat = ((alpha_t * x_t_cat + (1 - alpha_t) / x_t_cat.shape[1]) *
                        (alpha_bar_tm1 * x_0_cat + (1 - alpha_bar_tm1) / x_0_cat.shape[1]))
                                # Normalize π to get valid probabilities
                pi_cat = pi_cat / (pi_cat.sum(dim=1, keepdim=True) + eps)
                pi_cat = torch.clamp(pi_cat, min=min_prob)
                # Sample from the categorical distribution
                q_t_tm1_cat = torch.distributions.Categorical(probs=pi_cat)
                qs_true_cat[i].append(q_t_tm1_cat)

        chain_dkl_gauss = torch.stack([
            torch.distributions.kl.kl_divergence(q, p).sum(dim=1)
            for q, p in zip(qs_true_gauss, ps_gauss)
        ]).sum().mean()
        
        chain_dkl_cat = 0
        chain_dkls_cat = [[] for _ in x_0s_cat]
        for i, (q_true_cat, p_cat) in enumerate(zip(qs_true_cat, ps_cat)):
            chain_dkl_cat = 0
            for q_true_cat_t, p_cat_t in zip(q_true_cat, p_cat):
                chain_dkl_cat += torch.distributions.kl.kl_divergence(q_true_cat_t, p_cat_t)
            chain_dkls_cat[i].append(chain_dkl_cat)

        
        LOSS_GAUSS = terminal_dkl_gauss + chain_dkl_gauss + nll_gaussian
        return LOSS_GAUSS
        

    def count_params(self):
        return sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)

    def linear_beta_scheduler(self, timesteps):
        beta_start = 1e-4
        beta_end = 0.02
        return torch.linspace(beta_start, beta_end, timesteps, requires_grad=False)

    def cosine_scheduler(self, T, s=0.008, scale_factor=0.1):
        timesteps = np.linspace(0, T, T + 1, dtype=np.float64)
        
        # Calculate alpha_t based on the cosine schedule with a scaling factor
        f_t = np.cos(((timesteps / T + s) / (1 + s)) * np.pi / 2) ** 2
        f_t /= f_t[0]  # Normalize to start from 1
        
        # Apply a scale factor to make the schedule less aggressive
        alpha_t = np.clip(f_t * (1 - scale_factor) + scale_factor, 0, 1)

        # Calculate alpha_bar (cumulative product of alpha_t)
        alpha_bar = np.cumprod(alpha_t)

        # Calculate beta_t (noise magnitude at each timestep)
        beta_t = 1 - alpha_t
        
        return alpha_bar, alpha_t, beta_t
    
    def print_histo_step(self, analysed_data):
        analysed_data_flat = analysed_data.detach().cpu().numpy()
        plt.figure(figsize=(10, 6))
        plt.hist(analysed_data_flat, bins=50, density=True, alpha=0.7, color='b', edgecolor='black')
        plt.title('Histogram of Noisy Gaussian Samples')
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.show()
    
