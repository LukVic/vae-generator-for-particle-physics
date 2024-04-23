import sys
sys.path.append("/home/lucas/Documents/KYR/msc_thesis/vae-generator-for-particle-physics/analysis/support_scripts")

import torch
import numpy as np
import pandas as pd
import json
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from feature_transform import tan_to_angle


def data_gen(PATH_DATA, DATA_FILE, PATH_MODEL, PATH_JSON, TYPE, scaler, reaction, dataset):
    
    SAMPLING = 'generate' #regenerate
    print(dataset.shape)
    if reaction == 'bkg_all':
        SAMPLES_NUM = dataset.shape[0]
    else:
        SAMPLES_NUM = dataset.shape[0]
        
    with open(f"{PATH_JSON}", 'r') as json_file:
        conf_dict = json.load(json_file)
    gen_params = conf_dict["general"]
    
    model = torch.load(PATH_MODEL)
    features_used = ['taus_pt_0', 'MtLepMet', 'met_met', 'DRll01', 'MLepMet', 'minDeltaR_LJ_0', 'jets_pt_0', 'HT', 'HT_lep', 'total_charge']
    
    model.eval()
    latent_dimension = gen_params["latent_size"]
    
    data_array = np.empty((0, dataset.shape[1]), dtype=np.float32)
    
    with torch.no_grad():
        if TYPE == 'std' or TYPE == 'sym':
            latent_samples = generate_latents(SAMPLES_NUM,latent_dimension,SAMPLING,model,dataset)
            print(f'SIZE OF THE DATASET: {SAMPLES_NUM}')
            xhats_mu_gauss, xhats_std_gauss, xhats_bernoulli = model.decoder.decode(latent_samples.to('cuda'))
            px_gauss = torch.distributions.Normal(xhats_mu_gauss, xhats_std_gauss)
            xhat_gauss = px_gauss.sample()
            xhat_bernoulli = torch.bernoulli(xhats_bernoulli)
            xhats = torch.cat((xhat_gauss, xhat_bernoulli.view(-1,1)), dim=1)
            x_hats_denorm = scaler.inverse_transform(xhats.cpu().numpy())
            #x_hats_denorm = tan_to_angle(conf_dict['angle_convert']['indices'], torch.tensor(x_hats_denorm)).numpy()
            data_array = np.vstack((data_array, x_hats_denorm))      
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
        else:
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
    
    # Create a DataFrame from the NumPy array
    df_gen = pd.DataFrame(data_array,columns=features_used)
    bound = 0.5
    replacement_lower = -2.0
    replacement_upper = 2.0
    
    mask = (df_gen['total_charge'] < bound)
    df_gen.loc[mask, 'total_charge'] = replacement_lower
    df_gen.loc[~mask, 'total_charge'] = replacement_upper
    
    df_gen['y'] = reaction

    print("Processing completed.")
    print(f'{PATH_DATA}generated_{DATA_FILE}_E{gen_params["num_epochs"]}_S{SAMPLES_NUM}_{TYPE}.csv')
    df_gen.to_csv(f'{PATH_DATA}generated_{DATA_FILE}_E{gen_params["num_epochs"]}_S{SAMPLES_NUM}_{TYPE}.csv', index=False)


def generate_latents(samples_num, latent_dimension, sampling, model, dataset):
    if sampling == 'generate':
        return torch.randn(samples_num, latent_dimension)
    else:
        mu, std = model.encoder.encode(torch.tensor(dataset,dtype=torch.float32).to('cuda'))
        posterior = torch.distributions.Normal(mu, std)
        return posterior.sample()