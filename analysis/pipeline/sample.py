import sys
sys.path.append("/home/lucas/Documents/KYR/msc_thesis/vae-generator-for-particle-physics/analysis/support_scripts")

import time

import torch
import numpy as np
import pandas as pd
import json
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from feature_transform import tan_to_angle


def data_gen(PATH_DATA, DATA_FILE, PATH_MODEL, PATH_JSON, TYPE, scaler, reaction, dataset):
    
    
    # Chose if generate new samples of just regenerate the simulated ones
    SAMPLING = 'generate' #regenerate
    print(dataset.shape)

    SAMPLES_NUM = dataset.shape[0] # Size of the generated dataset
    #SAMPLES_NUM = 5000000
    
    
    with open(f"{PATH_JSON}", 'r') as json_file:
        conf_dict = json.load(json_file)
    gen_params = conf_dict["general"]
    
    model = torch.load(PATH_MODEL)
    features_used = ['taus_pt_0', 'MtLepMet', 'met_met', 'DRll01', 'MLepMet', 'minDeltaR_LJ_0', 'jets_pt_0', 'HT', 'HT_lep', 'total_charge']
    
    model.eval()
    latent_dimension = gen_params["latent_size"]
    
    data_array = np.empty((0, dataset.shape[1]), dtype=np.float32)
    
    
    batch_size = dataset.shape[0] # Number of samples by batch
    latent_samples = []
    
    start_time = time.time()
    
    with torch.no_grad():
        # Generate new samples with standard architectures
        if TYPE == 'std' or TYPE == 'sym':
            print(f'SIZE OF THE DATASET: {SAMPLES_NUM}')
            
            for i in range(0, SAMPLES_NUM, batch_size):
                latent_samples_batch = generate_latents(batch_size, latent_dimension, SAMPLING, model, dataset)
                latent_samples.append(latent_samples_batch)
            
            latent_samples = torch.cat(latent_samples, dim=0)

            latent_samples = latent_samples.to('cuda')
            
            decoded_samples = []
            for i in range(0, latent_samples.size(0), batch_size):
                print(i)
                xhats_mu_gauss, xhats_std_gauss, xhats_bernoulli = model.decoder.decode(latent_samples[i:i+batch_size])
                px_gauss = torch.distributions.Normal(xhats_mu_gauss, xhats_std_gauss)
                xhat_gauss = px_gauss.sample()
                xhat_bernoulli = torch.bernoulli(xhats_bernoulli)
                xhats = torch.cat((xhat_gauss, xhat_bernoulli.view(-1,1)), dim=1)
                x_hats_denorm = scaler.inverse_transform(xhats.cpu().numpy())
                decoded_samples.append(torch.tensor(x_hats_denorm))
            data_array =  torch.cat(decoded_samples, dim=0)
        # Generate new samples with Ladder ELBO
        # TODO: generation by batches 
        elif TYPE == 'std_h':
            z_2 = generate_latents(SAMPLES_NUM,latent_dimension,SAMPLING,model,dataset)
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
            z_2 = generate_latents(SAMPLES_NUM,latent_dimension,SAMPLING,model,dataset)
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
            prior_sample = generate_latents(SAMPLES_NUM,latent_dimension,SAMPLING,model,dataset)
            
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
    
    df_gen = pd.DataFrame(data_array,columns=features_used)
    # Adjust the values for total_charge variable
    print(set(df_gen['total_charge']))
    print((df_gen['total_charge'] > 0.5).sum())
    bound = 0.5
    replacement_lower = -2.0
    replacement_upper = 2.0
    mask = (df_gen['total_charge'] < bound)
    df_gen.loc[mask, 'total_charge'] = replacement_lower
    df_gen.loc[~mask, 'total_charge'] = replacement_upper
    
    df_gen['y'] = reaction

    end_time = time.time()
    print("Time taken:", end_time - start_time, "seconds")
    print("Processing completed.")
    print(f'{PATH_DATA}generated_{DATA_FILE}_E{gen_params["num_epochs"]}_S{SAMPLES_NUM}_{TYPE}.csv')
    df_gen.to_csv(f'{PATH_DATA}generated_{DATA_FILE}_E{gen_params["num_epochs"]}_S{SAMPLES_NUM}_{TYPE}.csv', index=False)

# Either only generates radom samples or encode the simulated dataset and sample from it 
def generate_latents(samples_num, latent_dimension, sampling, model, dataset):
    if sampling == 'generate':
        return torch.randn(samples_num, latent_dimension)
    else:
        mu, std = model.encoder.encode(torch.tensor(dataset,dtype=torch.float32).to('cuda'))
        posterior = torch.distributions.Normal(mu, std)
        return posterior.sample()