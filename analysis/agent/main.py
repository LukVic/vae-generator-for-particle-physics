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
from architecture import VAE
from applications import *

def main():
    OPTIMIZE = False
    PATH_JSON = '/home/lucas/Documents/KYR/msc_thesis/vae-generator-for-particle-physics/analysis/config/'
    PATH_DATA = '/home/lucas/Documents/KYR/msc_thesis/vae-generator-for-particle-physics/analysis/data/'
    PATH_MODEL = '/home/lucas/Documents/KYR/msc_thesis/vae-generator-for-particle-physics/analysis/models/'
    
    DATA_FILE = 'df_no_zeros'
    
    df = pd.read_csv(f'{PATH_DATA}{DATA_FILE}.csv')
    train_dataset = torch.tensor(df.values, dtype=torch.float32)
    
    if torch.cuda.is_available():
        print("CUDA (GPU) is available.")
        device = 'cuda'
    else:
        print("CUDA (GPU) is not available.")
        device = 'cpu'
    
    with open(f"{PATH_JSON}hyperparams.json", 'r') as json_file:
        conf_dict = json.load(json_file)
    
    gen_params = conf_dict["general"]
    
    if OPTIMIZE:
        #! HERE IS PERFORMED THE OPTIMIZATION
        pass
    
    input_size = train_dataset.shape[1]
    elbo_history = []
        
    scaler = StandardScaler()
    #scaler = MinMaxScaler()
    train_dataset_norm = scaler.fit_transform(train_dataset)
    train_dataloader = DataLoader(train_dataset_norm, batch_size=gen_params["batch_size"], shuffle=True)
    #train_dataset_norm[:,7] = np.round(train_dataset_norm[:,7]).astype(int)
    #print(train_dataset_norm[train_dataset_norm == 1].shape)
    #exit()

    # Create model and optimizer
    model = VAE(gen_params["latent_size"], device, input_size, conf_dict)
    optimizer = optim.Adam(model.parameters(), lr=gen_params["lr"])

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
            elbo_history.append(loss.item())
            progress_bar.set_description(f'EPOCH: {epoch+1}/{gen_params["num_epochs"]} | LOSS: {loss:.7f}')
            progress_bar.update(1)
        progress_bar.close()     

    torch.save(model, f'{PATH_MODEL}{DATA_FILE}_disc_{gen_params["num_epochs"]}.pth')
    
    #event_regen(input_size, scaler, train_dataloader, f'{PATH_MODEL}{DATA_FILE}_{gen_params["num_epochs"]}.pth')
    pos_collapse(train_dataloader, f'{PATH_MODEL}{DATA_FILE}_disc_{gen_params["num_epochs"]}.pth', f'{PATH_JSON}hyperparams.json')
    elbo_plot(elbo_history,f'{PATH_MODEL}{DATA_FILE}_disc_{gen_params["num_epochs"]}.pth', 'std')
    
if __name__ == "__main__":
    main()