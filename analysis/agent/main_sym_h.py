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
from architecture_sym_h import VAE
from applications import *
from dataset_new_h import dataset_regen_h

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
        #! HERE THE OPTIMIZATION IS PERFORMED
        pass
    
    input_size = train_dataset.shape[1]
    elbo_history1 = []
    elbo_history2 = []
    
    elbo_min1 = np.inf
    elbo_min2 = np.inf
        
    scaler = StandardScaler()
    #scaler = MinMaxScaler()
    train_dataset_norm = scaler.fit_transform(train_dataset)
    train_dataloader = DataLoader(train_dataset_norm, batch_size=gen_params["batch_size"], shuffle=True)

    # Create model and optimizer
    model = VAE(gen_params["latent_size"], device, input_size, conf_dict)
    optimizer = optim.Adam(model.parameters(), lr=gen_params["lr"])

    directory = f'sym_h_{gen_params["num_epochs"]}_epochs_model/'

    if not os.path.exists(f'{PATH_MODEL}{directory}'):
        os.makedirs(f'{PATH_MODEL}{directory}')

    # Train the model
    model.train()
    for epoch in range(gen_params["num_epochs"]):
        progress_bar = tqdm(total=len(train_dataloader))
        for _, x in enumerate(train_dataloader):
            
            #! THE FIRST STEP
            x = x.view(-1, input_size)
            optimizer.zero_grad()
            variables = model(x.float(), 1)
            loss1 = model.loss_function(x, variables, 1)
            loss1.backward()
            optimizer.step()
            
            #! THE SECOND STEP
            pz2_gauss = torch.distributions.Normal(torch.zeros((gen_params["batch_size"],gen_params["latent_size"])),
                                                  torch.ones((gen_params["batch_size"],gen_params["latent_size"])))
            z_2 = pz2_gauss.sample()
            xhats_mu_gauss_1, xhats_sigma_gauss_1 = model.encoder_3(z_2.to('cuda'))
            pz1z2 = torch.distributions.Normal(xhats_mu_gauss_1, torch.exp(xhats_sigma_gauss_1))
            z = pz1z2.sample()
            optimizer.zero_grad()
            distr_grad = model(z_2.float(), 2)
            loss2 = model.loss_function(z, distr_grad, 2)
            loss2.backward()
            optimizer.step()
            #exit()
            
            progress_bar.set_description(f'EPOCH: {epoch+1}/{gen_params["num_epochs"]} | LOSS: {loss2:.7f}')
            progress_bar.update(1)
        progress_bar.close()     
        elbo_history1.append(loss1.item())
        elbo_history2.append(loss2.item())
        if epoch % 1500 == 0:
            print(f'1: OLD/NEW: {elbo_min1}/{loss1.item()}')
            print(f'2: OLD/NEW: {elbo_min2}/{loss2.item()}') 

            torch.save(model, f'{PATH_MODEL}{directory}{DATA_FILE}_disc_{gen_params["num_epochs"]}_{epoch}_best.pth')
            elbo_plot(elbo_history1,f'{PATH_MODEL}{directory}{DATA_FILE}_disc_{gen_params["num_epochs"]}_{epoch}_best.pth', 'sym_1_h', f'{PATH_JSON}hyperparams.json')
            elbo_plot(elbo_history2,f'{PATH_MODEL}{directory}{DATA_FILE}_disc_{gen_params["num_epochs"]}_{epoch}_best.pth', 'sym_2_h', f'{PATH_JSON}hyperparams.json')
            dataset_regen_h(PATH_DATA, DATA_FILE, PATH_MODEL = f'{PATH_MODEL}{directory}{DATA_FILE}_disc_{gen_params["num_epochs"]}_{epoch}_best.pth', PATH_JSON=f'{PATH_JSON}hyperparams.json', EPOCHS=epoch, TYPE='sym_h', scaler=scaler)
        if (elbo_min1 > loss1.item() and elbo_min2 > loss2.item()):
            print("SAVING NEW BEST MODEL")
            torch.save(model, f'{PATH_MODEL}{directory}{DATA_FILE}_disc_best.pth')
            elbo_plot(elbo_history1,f'{PATH_MODEL}{directory}{DATA_FILE}_disc_best.pth', 'sym_1_h', f'{PATH_JSON}hyperparams.json')
            elbo_plot(elbo_history2,f'{PATH_MODEL}{directory}{DATA_FILE}_disc_best.pth', 'sym_2_h', f'{PATH_JSON}hyperparams.json')
            dataset_regen_h(PATH_DATA, DATA_FILE, PATH_MODEL = f'{PATH_MODEL}{directory}{DATA_FILE}_disc_best.pth', PATH_JSON=f'{PATH_JSON}hyperparams.json', EPOCHS=gen_params["num_epochs"], TYPE='sym_h', scaler=scaler)  
            elbo_min1 = loss1.item()
            elbo_min2 = loss2.item()

    # torch.save(model, f'{PATH_MODEL}{directory}{DATA_FILE}_disc_{gen_params["num_epochs"]}_{gen_params["num_epochs"]}.pth')
    # #! ADD POSTERIOR COLLAPS
    # #pos_collapse(train_dataloader, f'{PATH_MODEL}{directory}{DATA_FILE}_disc_{gen_params["num_epochs"]}.pth', f'{PATH_JSON}hyperparams.json')
    # elbo_plot(elbo_history1,f'{PATH_MODEL}{directory}{DATA_FILE}_disc_{gen_params["num_epochs"]}_{gen_params["num_epochs"]}.pth', 'sym_1_h', f'{PATH_JSON}hyperparams.json')
    # elbo_plot(elbo_history2,f'{PATH_MODEL}{directory}{DATA_FILE}_disc_{gen_params["num_epochs"]}_{gen_params["num_epochs"]}.pth', 'sym_2_h', f'{PATH_JSON}hyperparams.json')
    # dataset_regen(PATH_DATA, DATA_FILE, PATH_MODEL = f'{PATH_MODEL}{directory}{DATA_FILE}_disc_{gen_params["num_epochs"]}_{gen_params["num_epochs"]}_best.pth',PATH_JSON=f'{PATH_JSON}hyperparams.json', EPOCHS=gen_params["num_epochs"], TYPE='sym', scaler=scaler)
if __name__ == "__main__":
    main()