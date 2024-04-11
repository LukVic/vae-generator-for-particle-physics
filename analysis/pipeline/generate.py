import sys
sys.path.append("/home/lucas/Documents/KYR/msc_thesis/vae-generator-for-particle-physics/analysis/support_scripts")
sys.path.append("/home/lucas/Documents/KYR/msc_thesis/vae-generator-for-particle-physics/analysis/agent")
import os
import csv

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
from sample import data_gen

# def main():
#     TRAIN = True
#     #'tbh_all' 'tth' 'ttw' 'ttz' 'tt'
#     REACTION = 'bkg_all'
#     OPTIMIZE = False
#     PATH_JSON = f'/home/lucas/Documents/KYR/msc_thesis/vae-generator-for-particle-physics/analysis/config/'
#     PATH_DATA = f'/home/lucas/Documents/KYR/msc_thesis/vae-generator-for-particle-physics/analysis/data/{REACTION}_input/'
#     PATH_MODEL = f'/home/lucas/Documents/KYR/msc_thesis/vae-generator-for-particle-physics/analysis/models/production/{REACTION}_input/'
#     PATH_FEATURES = f'/home/lucas/Documents/KYR/msc_thesis/vae-generator-for-particle-physics/analysis/features/'
#     #classes = {'tbh_800': 0, 'tth' : 1, 'ttw': 1, 'ttz' : 1, 'tt' : 1}
#     classes = {'tbh_800_new': 0, 'bkg_all': 1}
#     #DATA_FILE = 'df_phi'
#     #DATA_FILE = 'df_no_zeros'
#     #DATA_FILE = 'df_8'
#     #DATA_FILE = 'df_pt'
#     DATA_FILE = f'df_{REACTION}_pres_strict'
#     FEATURES_FILE = f'features_top_10'
    
#     df = pd.read_csv(f'{PATH_DATA}{DATA_FILE}.csv')
#     df = df.drop(columns=['weight', 'row_number'])
    
#     features = []
    
#     with open(f'{PATH_FEATURES}{FEATURES_FILE}.csv', 'r') as file:
#         csv_reader = csv.reader(file)
#         for row in csv_reader:
#             features.append(row[0])
#     df = df[features]
    
#     train_dataset = torch.tensor(df.values, dtype=torch.float32)
#     # features = pd.read_csv(f'{PATH_FEATURES}{FEATURES_FILE}.csv')


#     with open(f"{PATH_JSON}hyperparams.json", 'r') as json_file:
#         conf_dict = json.load(json_file)
    
#     if torch.cuda.is_available():
#         print("CUDA (GPU) is available.")
#         device = 'cuda'
#     else:
#         print("CUDA (GPU) is not available.")
#         device = 'cpu'
        
#     gen_params = conf_dict["general"]
    
#     input_size = train_dataset.shape[1]
#     elbo_history = []
#     elbo_min = np.inf
        
#     scaler = StandardScaler()
#     #scaler = MinMaxScaler()
#     train_dataset_norm = scaler.fit_transform(train_dataset)
#     train_dataloader = DataLoader(train_dataset_norm, batch_size=gen_params["batch_size"], shuffle=True)

#     # Create model and optimizer
#     model = VAE(gen_params["latent_size"], device, input_size, conf_dict)
#     optimizer = optim.Adam(model.parameters(), lr=gen_params["lr"])

#     directory = f'std_{gen_params["num_epochs"]}_epochs_model/'

#     if not os.path.exists(f'{PATH_MODEL}{directory}'):
#         os.makedirs(f'{PATH_MODEL}{directory}')

#     if TRAIN:
#         # Train the model
#         model.train()
#         for epoch in range(gen_params["num_epochs"]):
#             progress_bar = tqdm(total=len(train_dataloader))
#             for _, x in enumerate(train_dataloader):
#                 x = x.view(-1, input_size)
#                 optimizer.zero_grad()
#                 x_hat, pz, qz, = model(x.float())
#                 loss = model.loss_function(x, x_hat, pz, qz)
#                 loss.backward()
#                 optimizer.step()
#                 progress_bar.set_description(f'EPOCH: {epoch+1}/{gen_params["num_epochs"]} | LOSS: {loss:.7f}')
#                 progress_bar.update(1)
#             progress_bar.close()     
#             elbo_history.append(loss.item())
#             if elbo_min > loss.item():
#                 print("SAVING NEW BEST MODEL")
#                 torch.save(model, f'{PATH_MODEL}{directory}{DATA_FILE}.pth')
#                 elbo_min = loss.item()
#         data_gen(PATH_DATA, DATA_FILE, PATH_MODEL = f'{PATH_MODEL}{directory}{DATA_FILE}.pth', PATH_JSON=f'{PATH_JSON}hyperparams.json', TYPE='std', scaler=scaler, reaction=classes[REACTION])
#     else:
#         model = torch.load(f'{PATH_MODEL}{directory}{DATA_FILE}.pth')
#         print(f'{PATH_MODEL}{directory}{DATA_FILE}.pth')
#         data_gen(PATH_DATA, DATA_FILE, PATH_MODEL = f'{PATH_MODEL}{directory}{DATA_FILE}.pth', PATH_JSON=f'{PATH_JSON}hyperparams.json', TYPE='std', scaler=scaler, reaction=classes[REACTION])

def main():
    TRAIN = True
    #'tbh_all' 'tth' 'ttw' 'ttz' 'tt'
    REACTION = 'bkg_all'
    OPTIMIZE = False
    PATH_JSON = f'/home/lucas/Documents/KYR/msc_thesis/vae-generator-for-particle-physics/analysis/config/'
    PATH_DATA = f'/home/lucas/Documents/KYR/msc_thesis/vae-generator-for-particle-physics/analysis/data/{REACTION}_input/'
    PATH_MODEL = f'/home/lucas/Documents/KYR/msc_thesis/vae-generator-for-particle-physics/analysis/models/production/{REACTION}_input/'
    PATH_FEATURES = f'/home/lucas/Documents/KYR/msc_thesis/vae-generator-for-particle-physics/analysis/features/'
    #classes = {'tbh_800': 0, 'tth' : 1, 'ttw': 1, 'ttz' : 1, 'tt' : 1}
    classes = {'tbh_800_new': 0, 'bkg_all': 1}
    #DATA_FILE = 'df_phi'
    #DATA_FILE = 'df_no_zeros'
    #DATA_FILE = 'df_8'
    #DATA_FILE = 'df_pt'
    DATA_FILE = f'df_{REACTION}_pres_strict'
    FEATURES_FILE = f'features_top_10'
    
    df = pd.read_csv(f'{PATH_DATA}{DATA_FILE}.csv')
    df = df.drop(columns=['weight', 'row_number'])
    
    features = []
    
    with open(f'{PATH_FEATURES}{FEATURES_FILE}.csv', 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            features.append(row[0])
    df = df[features]
    
    train_dataset = torch.tensor(df.values, dtype=torch.float32)
    # features = pd.read_csv(f'{PATH_FEATURES}{FEATURES_FILE}.csv')


    with open(f"{PATH_JSON}hyperparams.json", 'r') as json_file:
        conf_dict = json.load(json_file)
    
    if torch.cuda.is_available():
        print("CUDA (GPU) is available.")
        device = 'cuda'
    else:
        print("CUDA (GPU) is not available.")
        device = 'cpu'
    
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

    directory = f'sym_{gen_params["num_epochs"]}_epochs_model/'

    if not os.path.exists(f'{PATH_MODEL}{directory}'):
        os.makedirs(f'{PATH_MODEL}{directory}')

    if TRAIN:
        # Train the model
        model.train()
        for epoch in range(gen_params["num_epochs"]):
            progress_bar = tqdm(total=len(train_dataloader))
            for _, x in enumerate(train_dataloader):
                
                #! THE FIRST STEP
                x = x.view(-1, input_size)
                optimizer.zero_grad()
                distr_grad = model(x.float(), 1)
                loss1 = model.loss_function(x, distr_grad, 1)
                loss1.backward()
                optimizer.step()
                
                #! THE SECOND STEP
                pz_gauss = torch.distributions.Normal(torch.zeros((gen_params["batch_size"],gen_params["latent_size"])),
                                                    torch.ones((gen_params["batch_size"],gen_params["latent_size"])))
                z = pz_gauss.sample()
                #print(z.shape)
                optimizer.zero_grad()
                distr_grad = model(z.float(), 2)
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
            if (elbo_min1 > loss1.item() and elbo_min2 > loss2.item()):
                print("SAVING NEW BEST MODEL")
                torch.save(model, f'{PATH_MODEL}{directory}{DATA_FILE}_disc_best.pth') 
                elbo_min1 = loss1.item()
                elbo_min2 = loss2.item()
            data_gen(PATH_DATA, DATA_FILE, PATH_MODEL = f'{PATH_MODEL}{directory}{DATA_FILE}.pth', PATH_JSON=f'{PATH_JSON}hyperparams.json', TYPE='sym', scaler=scaler, reaction=classes[REACTION])
    else:
        model = torch.load(f'{PATH_MODEL}{directory}{DATA_FILE}.pth')
        print(f'{PATH_MODEL}{directory}{DATA_FILE}.pth')
        data_gen(PATH_DATA, DATA_FILE, PATH_MODEL = f'{PATH_MODEL}{directory}{DATA_FILE}.pth', PATH_JSON=f'{PATH_JSON}hyperparams.json', TYPE='sym', scaler=scaler, reaction=classes[REACTION])



if __name__ == "__main__":
    main()