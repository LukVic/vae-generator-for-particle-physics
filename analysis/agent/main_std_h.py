import sys
sys.path.append("/home/lucas/Documents/KYR/msc_thesis/vae-generator-for-particle-physics/analysis/support_scripts")
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
from architecture_std_h import VAE
from applications import *
from dataset_new_h import dataset_regen_h
from feature_transform import angle_to_tan

def main():
    OPTIMIZE = False
    PATH_JSON = '/home/lucas/Documents/KYR/msc_thesis/vae-generator-for-particle-physics/analysis/config/'
    PATH_DATA = '/home/lucas/Documents/KYR/msc_thesis/vae-generator-for-particle-physics/analysis/data/tt/'
    PATH_MODEL = '/home/lucas/Documents/KYR/msc_thesis/vae-generator-for-particle-physics/analysis/models/'
    
    #DATA_FILE = 'df_phi'
    DATA_FILE = 'df_no_zeros'
    #DATA_FILE = 'df_8'
    #DATA_FILE = 'df_pt'
    
    df = pd.read_csv(f'{PATH_DATA}{DATA_FILE}.csv')
    train_dataset = torch.tensor(df.values, dtype=torch.float32)

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
    
    if OPTIMIZE:
        #! HERE IS PERFORMED THE OPTIMIZATION
        pass
    
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

    directory = f'std_h_{gen_params["num_epochs"]}_epochs_model/'

    if not os.path.exists(f'{PATH_MODEL}{directory}'):
        os.makedirs(f'{PATH_MODEL}{directory}')

    # Train the model
    model.train()
    for epoch in range(gen_params["num_epochs"]):
        
        progress_bar = tqdm(total=len(train_dataloader))
        
        for _, x in enumerate(train_dataloader):
            x = x.view(-1, input_size)
            optimizer.zero_grad()
            x_hat, pz_1, pz_2, qz_1, qz_2 = model(x.float())
            loss = model.loss_function(x, x_hat, pz_1, pz_2, qz_1, qz_2)
            loss.backward()
            optimizer.step()
            progress_bar.set_description(f'EPOCH: {epoch+1}/{gen_params["num_epochs"]} | LOSS: {loss:.7f}')
            progress_bar.update(1)
        
        
        progress_bar.close()     
        elbo_history.append(loss.item())
        if epoch % 100 == 0 :
            print(f'OLD/NEW: {elbo_min}/{loss.item()}') 
            torch.save(model, f'{PATH_MODEL}{directory}{DATA_FILE}_disc_{gen_params["num_epochs"]}_{epoch}_best.pth')
            #pos_collapse(train_dataloader, f'{PATH_MODEL}{directory}{DATA_FILE}_disc_{gen_params["num_epochs"]}_{epoch}_best.pth', f'{PATH_JSON}hyperparams.json')
            elbo_plot(elbo_history,f'{PATH_MODEL}{directory}{DATA_FILE}_disc_{gen_params["num_epochs"]}_{epoch}_best.pth', 'std', f'{PATH_JSON}hyperparams.json')
            dataset_regen_h(PATH_DATA, DATA_FILE, PATH_MODEL = f'{PATH_MODEL}{directory}{DATA_FILE}_disc_{gen_params["num_epochs"]}_{epoch}_best.pth', PATH_JSON=f'{PATH_JSON}hyperparams.json', EPOCHS=epoch, TYPE='std_h', scaler=scaler)
        if elbo_min > loss.item():
            print("SAVING NEW BEST MODEL")
            torch.save(model, f'{PATH_MODEL}{directory}{DATA_FILE}_disc_best.pth')
            elbo_plot(elbo_history,f'{PATH_MODEL}{directory}{DATA_FILE}_disc_best.pth', 'std_h', f'{PATH_JSON}hyperparams.json')
            dataset_regen_h(PATH_DATA, DATA_FILE, PATH_MODEL = f'{PATH_MODEL}{directory}{DATA_FILE}_disc_best.pth', PATH_JSON=f'{PATH_JSON}hyperparams.json', EPOCHS=gen_params["num_epochs"], TYPE='std_h', scaler=scaler)
            elbo_min = loss.item()

    # torch.save(model, f'{PATH_MODEL}{directory}{DATA_FILE}_disc_{gen_params["num_epochs"]}_{gen_params["num_epochs"]}.pth')
    # #pos_collapse(train_dataloader, f'{PATH_MODEL}{directory}{DATA_FILE}_disc_{gen_params["num_epochs"]}.pth', f'{PATH_JSON}hyperparams.json')
    # elbo_plot(elbo_history,f'{PATH_MODEL}{directory}{DATA_FILE}_disc_{gen_params["num_epochs"]}_{gen_params["num_epochs"]}.pth', 'std', f'{PATH_JSON}hyperparams.json')
    # dataset_regen(PATH_DATA, DATA_FILE, PATH_MODEL = f'{PATH_MODEL}{directory}{DATA_FILE}_disc_{gen_params["num_epochs"]}_{gen_params["num_epochs"]}_best.pth',PATH_JSON=f'{PATH_JSON}hyperparams.json', EPOCHS=gen_params["num_epochs"], TYPE='std', scaler=scaler)
if __name__ == "__main__":
    main()