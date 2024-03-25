import sys
sys.path.append("/home/lucas/Documents/KYR/msc_thesis/vae-generator-for-particle-physics/analysis/support_scripts")
sys.path.append("/home/lucas/Documents/KYR/msc_thesis/vae-generator-for-particle-physics/analysis/agent")
import os

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from tqdm.auto import tqdm
import json

#from architecture import VAE
from architecture_std import VAE
from applications import *
from sample import data_gen

def main():
    TRAIN = False
    #'tbh_all' 'tth' 'ttw' 'ttz' 'tt'
    REACTION = 'tt'
    OPTIMIZE = False
    PATH_JSON = f'/home/lucas/Documents/KYR/msc_thesis/vae-generator-for-particle-physics/analysis/config/'
    PATH_DATA = f'/home/lucas/Documents/KYR/msc_thesis/vae-generator-for-particle-physics/analysis/data/{REACTION}_input/'
    PATH_MODEL = f'/home/lucas/Documents/KYR/msc_thesis/vae-generator-for-particle-physics/analysis/models/production/{REACTION}_input/'
    PATH_FEATURES = f'/home/lucas/Documents/KYR/msc_thesis/vae-generator-for-particle-physics/analysis/features/'
    classes = {'tbh_all': 0, 'tth' : 1, 'ttw': 2, 'ttz' : 3, 'tt' : 4}
    
    #DATA_FILE = 'df_phi'
    #DATA_FILE = 'df_no_zeros'
    #DATA_FILE = 'df_8'
    #DATA_FILE = 'df_pt'
    DATA_FILE = f'df_{REACTION}_full_vec_pres_loose_feature_cut'
    FEATURES_FILE = f'best_5'
    
    df = pd.read_csv(f'{PATH_DATA}{DATA_FILE}.csv')
    df = df.drop(columns=['weight', 'row_number'])
    train_dataset = torch.tensor(df.values, dtype=torch.float32)
    features = pd.read_csv(f'{PATH_FEATURES}{FEATURES_FILE}.csv')
    
    with open(f"{PATH_JSON}hyperparams.json", 'r') as json_file:
        conf_dict = json.load(json_file)

    #! convert angle to its tangens
    #train_dataset = angle_to_tan(conf_dict['angle_convert']['indices'], train_dataset)
    #print(train_dataset.T[9])
    
    if torch.cuda.is_available():
        print("CUDA (GPU) is available.")
        device = 'cuda'
    else:
        print("CUDA (GPU) is not available.")
        device = 'cpu'
        
    gen_params = conf_dict["general"]
    
    input_size = train_dataset.shape[1]
    elbo_history = []
    elbo_min = np.inf
        
    scaler = StandardScaler()
    #scaler = MinMaxScaler()
    train_dataset_norm = scaler.fit_transform(train_dataset)
    train_dataloader = DataLoader(train_dataset_norm, batch_size=gen_params["batch_size"], shuffle=True)

    # Create model and optimizer
    model = VAE(gen_params["latent_size"], device, input_size, conf_dict)
    optimizer = optim.Adam(model.parameters(), lr=gen_params["lr"])

    directory = f'std_{gen_params["num_epochs"]}_epochs_model/'

    if not os.path.exists(f'{PATH_MODEL}{directory}'):
        os.makedirs(f'{PATH_MODEL}{directory}')

    if TRAIN:
        # Train the model
        model.train()
        for epoch in range(gen_params["num_epochs"]):
            progress_bar = tqdm(total=len(train_dataloader))
            for _, x in enumerate(train_dataloader):
                x = x.view(-1, input_size)
                optimizer.zero_grad()
                x_hat, pz, qz, = model(x.float())
                loss = model.loss_function(x, x_hat, pz, qz)
                loss.backward()
                optimizer.step()
                progress_bar.set_description(f'EPOCH: {epoch+1}/{gen_params["num_epochs"]} | LOSS: {loss:.7f}')
                progress_bar.update(1)
            progress_bar.close()     
            elbo_history.append(loss.item())
            if elbo_min > loss.item():
                print("SAVING NEW BEST MODEL")
                torch.save(model, f'{PATH_MODEL}{directory}{DATA_FILE}.pth')
                data_gen(PATH_DATA, DATA_FILE, PATH_MODEL = f'{PATH_MODEL}{directory}{DATA_FILE}.pth', PATH_JSON=f'{PATH_JSON}hyperparams.json', TYPE='std', scaler=scaler, reaction=classes[REACTION])
                elbo_min = loss.item()
    else:
        model = torch.load(f'{PATH_MODEL}{directory}{DATA_FILE}.pth')
        data_gen(PATH_DATA, DATA_FILE, PATH_MODEL = f'{PATH_MODEL}{directory}{DATA_FILE}.pth', PATH_JSON=f'{PATH_JSON}hyperparams.json', TYPE='std', scaler=scaler, reaction=classes[REACTION])
if __name__ == "__main__":
    main()