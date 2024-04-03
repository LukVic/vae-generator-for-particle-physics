import sys
sys.path.append("/home/lucas/Documents/KYR/msc_thesis/vae-generator-for-particle-physics/analysis/support_scripts")

import torch
import numpy as np
import pandas as pd
import json
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from feature_transform import tan_to_angle


def data_gen(PATH_DATA, DATA_FILE, PATH_MODEL, PATH_JSON, TYPE, scaler, reaction):
    
    SAMPLES = 27611
    with open(f"{PATH_JSON}", 'r') as json_file:
        conf_dict = json.load(json_file)
    gen_params = conf_dict["general"]
    
    model = torch.load(PATH_MODEL)
    features_used = ['taus_pt_0', 'MtLepMet', 'met_met', 'DRll01', 'MLepMet', 'minDeltaR_LJ_0', 'jets_pt_0', 'HT', 'HT_lep', 'total_charge']
    
    
    input_size = len(features_used)
    
    model.eval()
    latent_dimension = gen_params["latent_size"]
    
    data_array = np.empty((0, input_size), dtype=np.float32)

    with torch.no_grad():
        latent_samples = torch.randn(SAMPLES, latent_dimension)
        print(f'SIZE OF THE DATASET: {SAMPLES}')
        xhats_mu_gauss, xhats_sigma_gauss, xhats_bernoulli = model.decoder(latent_samples.to('cuda'))
        px_gauss = torch.distributions.Normal(xhats_mu_gauss, torch.exp(xhats_sigma_gauss))
        xhat_gauss = px_gauss.sample()
        xhat_bernoulli = torch.bernoulli(xhats_bernoulli)
        xhats = torch.cat((xhat_gauss, xhat_bernoulli.view(-1,1)), dim=1)
        x_hats_denorm = scaler.inverse_transform(xhats.cpu().numpy())
        #x_hats_denorm = tan_to_angle(conf_dict['angle_convert']['indices'], torch.tensor(x_hats_denorm)).numpy()
        data_array = np.vstack((data_array, x_hats_denorm))

    # Create a DataFrame from the NumPy array
    df_gen = pd.DataFrame(data_array,columns=features_used)
    lower_bound = 0.2
    upper_bound = 0.3
    replacement_lower = -2.0
    replacement_upper = 2.0
    
    mask = (df_gen['total_charge'] >= lower_bound) & (df_gen['total_charge'] <= upper_bound)
    df_gen.loc[mask, 'total_charge'] = replacement_lower
    df_gen.loc[~mask, 'total_charge'] = replacement_upper
    
    df_gen['y'] = reaction

    print("Processing completed.")

    df_gen.to_csv(f'{PATH_DATA}generated_{DATA_FILE}_E{gen_params["num_epochs"]}_S{SAMPLES}_{TYPE}.csv', index=False)
