import sys
sys.path.append("/home/lucas/Documents/KYR/msc_thesis/vae-generator-for-particle-physics/analysis/support_scripts")

import torch
import numpy as np
import pandas as pd
import json
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from feature_transform import tan_to_angle


def dataset_regen_h(PATH_DATA, DATA_FILE, PATH_MODEL, PATH_JSON, EPOCHS, TYPE, scaler):
    with open(f"{PATH_JSON}", 'r') as json_file:
        conf_dict = json.load(json_file)
    gen_params = conf_dict["general"]
    
    model = torch.load(PATH_MODEL)
    df_real = pd.read_csv(f'{PATH_DATA}{DATA_FILE}.csv')
    
    train_dataset = torch.tensor(df_real.values, dtype=torch.float32)
    
    input_size = train_dataset.shape[1]
    
    model.eval()
    latent_dimension = gen_params["latent_size"]
    
    data_array = np.empty((0, input_size), dtype=np.float32)

    with torch.no_grad():
        z_2 = torch.randn(train_dataset.shape[0], latent_dimension)
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

    df_regen = pd.DataFrame(data_array,columns=df_real.columns)

    print("Processing completed.")

    df_regen.to_csv(f'{PATH_DATA}{DATA_FILE}_disc_{gen_params["num_epochs"]}_{EPOCHS}_{TYPE}.csv', index=False)

def main():
    TYPE = 'std'
    EPOCHS = 1000
    PATH_MODEL = f'/home/lucas/Documents/KYR/msc_thesis/vae-generator-for-particle-physics/analysis/models/{TYPE}_{EPOCHS}_epochs_model/'
    PATH_DATA = '/home/lucas/Documents/KYR/msc_thesis/vae-generator-for-particle-physics/analysis/data/'
    DATA_FILE = 'df_phi'
    
    scaler = StandardScaler()
    #scaler = MinMaxScaler()
    dataset_regen_h(PATH_DATA, DATA_FILE, PATH_MODEL, EPOCHS, TYPE, scaler)
    
if __name__ == "__main__":
    main()