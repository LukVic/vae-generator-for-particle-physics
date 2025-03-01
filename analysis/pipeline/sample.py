import sys
sys.path.append("/home/lucas/Documents/KYR/msc_thesis/vae-generator-for-particle-physics/analysis/support_scripts")

import time

import torch
import numpy as np
import pandas as pd
import copy

from dataloader import load_config
from gm_helpers import infer_feature_type
from aux_functions import compute_embed, visualize_embed
import torch.nn.functional as F
import matplotlib.pyplot as plt


def data_gen(PATH_DATA, DATA_FILE, PATH_MODEL, PATH_JSON, TYPE, scaler, reaction, dataset_original, dataset_one_hot, feature_type_dict, features_list, prior):
    # Chose if generate new samples of just regenerate the simulated ones
    SAMPLING = 'generate' #regenerate

    SAMPLES_NUM = dataset_one_hot.shape[0] # Size of the generated dataset
    #SAMPLES_NUM = 5000000
    batch_size = SAMPLES_NUM # Number of samples by batch
    
    gen_params = load_config(PATH_JSON)['generate']['general']
    latent_dimension = gen_params["latent_size"]
    
    model = torch.load(PATH_MODEL)
    model.eval()
    
    data_array = np.empty((0, dataset_one_hot.shape[1]), dtype=np.float32)
    prior_samples = []
    posterior_samples = []
    z_2_samples = []
    
    start_time = time.time()
    
    with torch.no_grad():
        # Generate new samples with standard architectures
        if TYPE == 'vae_std' or TYPE == 'vae_sym':
            print(f'SIZE OF THE DATASET: {SAMPLES_NUM}')
            
            for i in range(0, SAMPLES_NUM, batch_size):
                prior_samples_batch, posterior_samples_batch = generate_latents(prior, batch_size, latent_dimension, SAMPLING, model, dataset_one_hot.values)
                prior_samples.append(prior_samples_batch)
                posterior_samples.append(posterior_samples_batch)
            
            prior_samples = torch.cat(prior_samples, dim=0).to('cuda')
            posterior_samples = torch.cat(posterior_samples, dim=0).to('cuda')
            
            decoded_samples = []
            for i in range(0, prior_samples.size(0), batch_size):
                xhats_mu_gauss, xhats_std_gauss, xhats_bernoulli, xhats_categorical = model.decoder(prior_samples[i:i+batch_size], feature_type_dict)

                px_gauss = torch.distributions.Normal(xhats_mu_gauss, xhats_std_gauss)
                xhats_gauss = px_gauss.sample()
                
                xhats_gauss_denorm = torch.tensor(scaler.inverse_transform(xhats_gauss.to('cpu')))
                xhats_bernoulli = torch.bernoulli(xhats_bernoulli)
                xhats = torch.cat((xhats_gauss_denorm, xhats_bernoulli.view(-1,len(feature_type_dict['binary_param'])).to('cpu')), dim=1)
                
                
                categorical_samples = []
                for start, end in feature_type_dict['categorical_only']:
                    categorical_distribution = torch.distributions.Categorical(logits=xhats_categorical[:, start:end])
                    xhat_categorical = categorical_distribution.sample().unsqueeze(1)
                    categorical_samples.append(xhat_categorical)

                xhats = torch.cat((xhats, torch.cat(categorical_samples, dim=1).to('cpu')), dim=1)
                
            data_array = xhats
            print(prior_samples[:,:len(features_list)].shape)
            print(data_array.shape)
            exit()
            compute_embed(prior_samples[:,:len(features_list)].to('cpu'),posterior_samples[:,:len(features_list)].to('cpu'),TYPE,'latent')
            visualize_embed(TYPE,'latent')
        # Generate new samples with Ladder ELBO
        # TODO: generation by batches 
        elif TYPE == 'lvae_std':
            
            print(f'SIZE OF THE DATASET: {SAMPLES_NUM}')
            
            for i in range(0, SAMPLES_NUM, batch_size):
                z_2_samples_batch = torch.randn(batch_size, latent_dimension)
                z_2_samples.append(z_2_samples_batch)
            
            z_2_samples = torch.cat(z_2_samples, dim=0).to('cuda')
            
            for i in range(0, z_2_samples.size(0), batch_size):
                
                xhats_mu_gauss_1, xhats_sigma_gauss_1 = model.encoder_3(z_2_samples[i:i+batch_size].to('cuda'))
                pz2z1 = torch.distributions.Normal(xhats_mu_gauss_1, xhats_sigma_gauss_1)
                z_1_samples = pz2z1.rsample()
                
                
                xhats_mu_gauss, xhats_std_gauss, xhats_bernoulli, xhats_categorical = model.decoder(z_1_samples[i:i+batch_size], feature_type_dict)

                px_gauss = torch.distributions.Normal(xhats_mu_gauss, xhats_std_gauss)
                xhats_gauss = px_gauss.sample()
                
                xhats_gauss_denorm = torch.tensor(scaler.inverse_transform(xhats_gauss.to('cpu')))
                xhats_bernoulli = torch.bernoulli(xhats_bernoulli)
                xhats = torch.cat((xhats_gauss_denorm, xhats_bernoulli.view(-1,len(feature_type_dict['binary_param'])).to('cpu')), dim=1)
                
                
                categorical_samples = []
                for start, end in feature_type_dict['categorical_only']:
                    categorical_distribution = torch.distributions.Categorical(logits=xhats_categorical[:, start:end])
                    xhat_categorical = categorical_distribution.sample().unsqueeze(1)
                    categorical_samples.append(xhat_categorical)

                xhats = torch.cat((xhats, torch.cat(categorical_samples, dim=1).to('cpu')), dim=1)
                
            data_array = xhats
            print(z_2_samples[:,:len(features_list)].shape)
            print(data_array.shape)
            compute_embed(z_2_samples[:,:len(features_list)].to('cpu'),z_1_samples[:,:len(features_list)].to('cpu'),TYPE,'ladder_latent')
            visualize_embed(TYPE,'ladder_latent')
            
        # TODO: generation by batches
        elif TYPE == 'lvae_sym':
            z_2 = generate_latents(SAMPLES_NUM,latent_dimension,SAMPLING,model,scaler.fit_transform(dataset_one_hot.values))
            print(f'SIZE OF THE DATASET: {SAMPLES_NUM}')
            # xhats_mu_gauss_3, xhats_sigma_gauss_3 = model.decoders[0].encode(z_3.to('cuda'))
            # pz3z2 = torch.distributions.Normal(xhats_mu_gauss_3, xhats_sigma_gauss_3)
            # z_2 = pz3z2.sample()
            xhats_mu_gauss_2, xhats_sigma_gauss_2 = model.decoders[0].encode(z_2.to('cuda'))
            pz2z1 = torch.distributions.Normal(xhats_mu_gauss_2, xhats_sigma_gauss_2)
            z_1 = pz2z1.sample()
            xhats_mu_gauss, xhats_sigma_gauss, xhats_bernoulli = model.decoders[1].decode(z_1)
            px_gauss = torch.distributions.Normal(xhats_mu_gauss, xhats_sigma_gauss)
            xhat_gauss = px_gauss.sample()
            xhat_bernoulli = torch.bernoulli(xhats_bernoulli)
            xhats = torch.cat((xhat_gauss, xhat_bernoulli.view(-1,1)), dim=1)
            x_hats_denorm = scaler.inverse_transform(xhats.cpu().numpy())
            #x_hats_denorm = tan_to_angle(conf_dict['angle_convert']['indices'], torch.tensor(x_hats_denorm)).numpy()
            data_array = np.vstack((data_array, x_hats_denorm))
        # Generate new samples with Deep diffusion generative model
        elif TYPE == 'ddgm':
            alpha_bars, alphas, betas = model.cosine_scheduler(model.timesteps)
            alpha_bars = torch.FloatTensor(alpha_bars).to(model.device)
            alphas = torch.FloatTensor(alphas).to(model.device)
            betas = torch.FloatTensor(betas).to(model.device)
            eps = 1e-8  # Small value for numerical stability
            min_prob = 1e-6  # Ensures no exact zeros 
            print(f'SIZE OF THE DATASET: {SAMPLES_NUM}')
            one_hot_ranges = [val[1]-val[0]+1 for val in feature_type_dict['binary_param']] + [val[1]-val[0] for val in feature_type_dict['categorical_one_hot']]

            prior_gauss = torch.distributions.Normal(torch.zeros(SAMPLES_NUM,len(feature_type_dict['real_data'])), torch.ones(SAMPLES_NUM,len(feature_type_dict['real_data'])))
            zs_gauss = prior_gauss.sample()

            zs_cat = []
            for one_hot_feature in one_hot_ranges:
                prior_cat = torch.distributions.Categorical(probs=torch.full((SAMPLES_NUM, one_hot_feature), 1.0 / one_hot_feature))
                prior_sample = prior_cat.sample()
                prior_sample_one_hot = F.one_hot(prior_sample, num_classes=one_hot_feature)
                zs_cat.append(prior_sample_one_hot)
            
            print(zs_cat)
            x_t = None
            x_t_gauss = zs_gauss.clone().to(model.device)
            x_ts_cat = copy.deepcopy(zs_cat)
            x_ts_cat = [x_t_cat.to(model.device) for x_t_cat in x_ts_cat]
            x_t = torch.cat([x_t_gauss,torch.cat(x_ts_cat,dim=1)],dim=1)
            for t, decoder in zip(reversed(range(1, model.timesteps)),reversed(model.decoders)):
                x_t_gauss_noise, x_0_cat_hat = decoder(x_t,feature_type_dict)
                
                # Retrieve precomputed values from the scheduler              
                alpha_bar_tm1 = alpha_bars[t-1].to(model.device)
                alpha_bar_t = alpha_bars[t].to(model.device) 
                alpha_t = alphas[t].to(model.device)
                beta_t = betas[t].to(model.device)
                
                # Gaussian Backward Path (for numerical features)
                mu_theta = (1 / torch.sqrt(alpha_t)) * (x_t_gauss - ((beta_t) / torch.sqrt(1 - alpha_bar_t)) * x_t_gauss_noise)
                sigma_t = torch.sqrt(beta_t) if t > 0 else 0.0  # No noise at final step
                q_t_tm1_gauss = torch.distributions.Normal(loc=mu_theta, scale=sigma_t)
                x_t_gauss = q_t_tm1_gauss.sample()
                print(t)
                for i, x_t_cat in enumerate(x_ts_cat): 
                    x_t_cat = x_t_cat.to(model.device)
                    pi_cat = ((alpha_t * x_t_cat + (1 - alpha_t) / x_t_cat.shape[1]) *
                            (alpha_bar_tm1 * x_0_cat_hat[i] + (1 - alpha_bar_tm1) / x_0_cat_hat[i].shape[1]))
                    
                    # Normalize π to get valid probabilities
                    pi_cat = pi_cat / (pi_cat.sum(dim=1, keepdim=True) + eps)
                    pi_cat = torch.clamp(pi_cat, min=min_prob)
                    # Sample from the categorical distribution
                    q_t_tm1_cat = torch.distributions.Categorical(probs=pi_cat)
                    x_t_features = q_t_tm1_cat.sample()
                    x_t_cat = F.one_hot(x_t_features.long(), num_classes=(one_hot_ranges[i]))
                    if t != 1: x_ts_cat[i] = x_t_cat
                    else: x_ts_cat[i] = x_t_features
                # print(x_t_gauss)
                # plt.figure(figsize=(10, 6))
                # plt.hist(x_t_gauss[:,0].cpu(), bins=50, density=True, alpha=0.7, color='b', edgecolor='black')
                # plt.title('Histogram of Noisy Gaussian Samples')
                # plt.xlabel('Value')
                # plt.ylabel('Density')
                # plt.show()
                if t != 1:
                    x_t = torch.cat([x_t_gauss, torch.cat(x_ts_cat, dim=1)], dim=1)
                else:
                    x_ts_cat = [x_t_cat.unsqueeze(1) for x_t_cat in x_ts_cat] 
                    x_t_cat = torch.cat(x_ts_cat, dim=1).cpu()
                    x_t_gauss_denorm = torch.tensor(scaler.inverse_transform(x_t_gauss.cpu())) 
                    x_t = torch.cat([x_t_gauss_denorm, x_t_cat], dim=1)
            data_array = x_t.cpu().numpy()
        
        elif TYPE == 'gan_std' or TYPE == 'wgan_gp':
            print(f'SIZE OF THE DATASET: {SAMPLES_NUM}')
            prior_samples_batch = torch.randn(SAMPLES_NUM, latent_dimension)
            for i in range(0, SAMPLES_NUM, batch_size):
                prior_samples.append(prior_samples_batch)

            prior_samples = torch.cat(prior_samples, dim=0).to('cuda')
            
            xhats = []
            for i in range(0, prior_samples.size(0), batch_size):
                xhat = model.generator(prior_samples[i:i+batch_size], feature_type_dict)
                print(feature_type_dict)
                # pick out the binary ones
                xhat_binary = xhat[:,feature_type_dict['binary_data'][0]:feature_type_dict['binary_data'][-1]+1]            
                xhat_binary = torch.sigmoid(xhat_binary)
                xhat_binary[xhat_binary>0.5] = 1
                xhat_binary[xhat_binary<=0.5] = 0
                # one-hot back to categorical
                xhats_unified = []
                for feature in feature_type_dict['categorical_one_hot']:
                    xhat_category = xhat[:, feature[0]:feature[1]]
                    probabilities = torch.softmax(xhat_category, dim=-1)
                    xhat_unified = torch.argmax(probabilities, dim=-1).unsqueeze(1)
                    xhats_unified.append(xhat_unified)
                xhats_unified = torch.cat(xhats_unified, dim=1)
                
                # denormalize real-valued features
                xhat_denorm = torch.tensor(scaler.inverse_transform(xhat[:,feature_type_dict['real_data'][0]:feature_type_dict['real_data'][-1]+1].to('cpu'))) # invert only real values fetures
                # concatenate the real-valued and the rest
                print(xhat_denorm.shape)
                print(xhat_binary.shape)
                print(xhats_unified.shape)
                print(xhat_denorm.to(torch.int))
                print(torch.round(xhat_denorm).int())

                xhat = torch.cat((xhat_denorm.to('cpu'), xhat_binary.to('cpu'), xhats_unified.to('cpu')), dim=1)
                xhats.append(xhat)
            # exit()
            data_array = torch.cat(xhats, dim=0)
            print(type(dataset_original))
            print(type(data_array))
            # compute_embed(data_array[:,:len(features_list)].to('cpu'),torch.tensor(dataset_original.values)[:,:len(features_list)].to('cpu'),TYPE,'data')
            # visualize_embed(TYPE,'data')
    print(features_list)
    print(data_array.shape)
    df_gen = pd.DataFrame(data_array,columns=features_list)
    
    df_gen = map_hist_vals(dataset_original,df_gen,feature_type_dict)
    
    df_gen['y'] = reaction

    end_time = time.time()
    print("Time taken:", end_time - start_time, "seconds")
    print("Processing completed.")
    print(f'{PATH_DATA}generated_{DATA_FILE}_E{gen_params["num_epochs"]}_S{SAMPLES_NUM}_{TYPE}.csv')
    df_gen.to_csv(f'{PATH_DATA}generated_{DATA_FILE}_E{gen_params["num_epochs"]}_S{SAMPLES_NUM}_{TYPE}.csv', index=False)

# Either only generates radom samples or encode the simulated dataset and sample from it 
#! RETURN BOTH TO VISUALIZE
def generate_latents(prior, samples_num, latent_dimension, sampling, model, dataset):
    #return torch.randn(samples_num, latent_dimension)
    mu, std = model.encoder(torch.tensor(dataset,dtype=torch.float32).to('cuda'))
    posterior = torch.distributions.Normal(mu, std)
    return prior.sample(samples_num), posterior.sample()

def map_hist_vals(df_simulated, df_generated, feature_type_dict):
    for idx in feature_type_dict['categorical_data']:
        hist_dict = dict(zip(sorted(df_generated.iloc[:,idx].unique()), sorted(df_simulated.iloc[:,idx].unique())))
        df_generated.iloc[:,idx] = df_generated.iloc[:,idx].map(hist_dict)
    for idx in feature_type_dict['binary_data']:
        hist_dict = dict(zip(sorted(df_generated.iloc[:,idx].unique()), sorted(df_simulated.iloc[:,idx].unique())))
        df_generated.iloc[:,idx] = df_generated.iloc[:,idx].map(hist_dict)
    return df_generated