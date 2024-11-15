import sys
sys.path.append("/home/lucas/Documents/KYR/msc_thesis/vae-generator-for-particle-physics/analysis/support_scripts")

import time

import torch
import numpy as np
import pandas as pd

from dataloader import load_config
from gm_helpers import infer_feature_type
from aux_functions import compute_embed, visualize_embed

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
    
    start_time = time.time()
    with torch.no_grad():
        # Generate new samples with standard architectures
        if TYPE == 'std' or TYPE == 'sym':
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
            compute_embed(prior_samples[:,:len(features_list)].to('cpu'),posterior_samples[:,:len(features_list)].to('cpu'),TYPE,'latent')
            visualize_embed(TYPE,'latent')
        # Generate new samples with Ladder ELBO
        # TODO: generation by batches 
        elif TYPE == 'std_h':
            z_2 = generate_latents(SAMPLES_NUM,latent_dimension,SAMPLING,model,scaler.fit_transform(dataset_one_hot.values))
            print(f'SIZE OF THE DATASET: {SAMPLES_NUM}')
            xhats_mu_gauss_1, xhats_sigma_gauss_1 = model.encoder_3(z_2.to('cuda'))
            pz2z1 = torch.distributions.Normal(xhats_mu_gauss_1, torch.exp(xhats_sigma_gauss_1))
            z_1 = pz2z1.rsample()
            xhats_mu_gauss, xhats_sigma_gauss, xhats_bernoulli = model.decoder(z_1)
            px_gauss = torch.distributions.Normal(xhats_mu_gauss, torch.exp(xhats_sigma_gauss))
            xhat_gauss = px_gauss.sample()
            xhat_bernoulli = torch.bernoulli(xhats_bernoulli)
            xhats = torch.cat((xhat_gauss, xhat_bernoulli.view(-1,1)), dim=1)
            x_hats_denorm = scaler.inverse_transform(xhats.cpu().numpy())
            #x_hats_denorm = tan_to_angle(conf_dict['angle_convert']['indices'], torch.tensor(x_hats_denorm)).numpy()
            data_array = np.vstack((data_array, x_hats_denorm))
        # Generate new samples with Ladder SEL
        # TODO: generation by batches
        elif TYPE == 'sym_h':
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
            prior_sample = generate_latents(SAMPLES_NUM,latent_dimension,SAMPLING,model,dataset_one_hot)
            
            z_current = prior_sample
            
            for encoder in reversed(model.encoders):
                z_mu, z_sigma = encoder(z_current.to('cuda'))
                distr = torch.distributions.Normal(z_mu, torch.exp(z_sigma))
                z_current = distr.sample()
            
            x_mu, x_sigma, x_p = model.decoder(z_current.to('cuda'))
            px = torch.distributions.Normal(x_mu, torch.exp(x_sigma))
            xhat_real = px.sample()
            xhat_bin = torch.bernoulli(x_p)
            xhats = torch.cat((xhat_real, xhat_bin.view(-1,1)), dim=1)
            x_hats_denorm = scaler.inverse_transform(xhats.cpu().numpy())
            data_array = np.vstack((data_array, x_hats_denorm))
        
        elif TYPE == 'gan':
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
                # one-hot back to categorical
                xhats_unified = []
                for feature in feature_type_dict['categorical_one_hot']:
                    xhat_category = xhat[:, feature[0]:feature[1]]
                    xhat_unified = xhat_category.argmax(dim=-1).unsqueeze(1)
                    xhats_unified.append(xhat_unified)
                xhats_unified = torch.cat(xhats_unified, dim=1)
                # denormalize real-valued features
                xhat_denorm = torch.tensor(scaler.inverse_transform(xhat[:,feature_type_dict['real_data'][0]:feature_type_dict['real_data'][-1]+1].to('cpu'))) # invert only real values fetures
                # concatenate the real-valued and the rest
                print(xhat_denorm.shape)
                print(xhat_binary.shape)
                print(xhats_unified.shape)
                xhat = torch.cat((xhat_denorm.to('cpu'), xhat_binary.to('cpu'), xhats_unified.to('cpu')), dim=1)
                xhats.append(xhat)
                
            data_array = torch.cat(xhats, dim=0)
            #compute_embed(xhats[:,:len(features_list)].to('cpu'),posterior_samples[:,:len(features_list)].to('cpu'),TYPE,'latent')
            #visualize_embed(TYPE,'latent')
    print(features_list)
    df_gen = pd.DataFrame(data_array,columns=features_list)
    # Adjust the values for total_charge variable
    #print(set(df_gen['total_charge']))
    #print((df_gen['total_charge'] > 0.5).sum())
    
    df_gen = map_hist_vals(dataset_original,df_gen,feature_type_dict)
    
    # bound = 0.5
    # replacement_lower = -2.0
    # replacement_upper = 2.0
    # mask = (df_gen['total_charge'] < bound)
    # df_gen.loc[mask, 'total_charge'] = replacement_lower
    # df_gen.loc[~mask, 'total_charge'] = replacement_upper
    
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