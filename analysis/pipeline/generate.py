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

from sample import data_gen

import matplotlib.pyplot as plt
def main():        
    #classes = {'tbh_800': 0, 'tth' : 1, 'ttw': 1, 'ttz' : 1, 'tt' : 1} # multinomimal classification dict
    classes = {'tbh_800_new': 0, 'bkg_all': 1} # binary classification dict
    
    PATH_JSON = f'../config/' # config path
    # load the general parameters
    with open(f"{PATH_JSON}hyperparams_gen.json", 'r') as json_file:
        conf_dict = json.load(json_file)
    gen_params = conf_dict["general"]
    
    TRAIN = gen_params["train"] # train if True, load existing model if False
    #'tbh_all' 'tth' 'ttw' 'ttz' 'tt'
    REACTION = gen_params["reaction"] # 'tbh_800_new' # signal or background
    
    PATH_DATA = f'../data/{REACTION}_input/' # training data path
    PATH_MODEL = f'../models/production/{REACTION}_input/' # model path
    PATH_FEATURES = f'../features/' # feature directory path

    DATA_FILE = f'df_{REACTION}_pres_strict' # data file based on demanded preselection
    FEATURES_FILE = f'features_top_10' # 'df_phi', 'df_no_zeros', 'df_8', 'df_pt'  feature file
    
        # which model to train
    APPROACH = gen_params["approach"] # 'std', 'sym', 'std_h', 'sym_h', 'ddgm'
    
    if APPROACH == 'std' or APPROACH == 'std_h' or APPROACH == 'ddgm':
        elbo_history = []
        elbo_min = np.inf
        if APPROACH == 'std': from architecture_std import VAE
        elif APPROACH == 'std_h': from architecture_std_h import VAE
        elif APPROACH == 'ddgm': from architecture_ddgm_simple import DDGM
        
    elif APPROACH == 'sym' or APPROACH == 'sym_h':
        elbo_history1 = []
        elbo_history2 = []
        elbo_min1 = np.inf
        elbo_min2 = np.inf
        if APPROACH == 'sym': from architecture_sym import VAE
        else: from architecture_sym_h import VAE

    df = pd.read_csv(f'{PATH_DATA}{DATA_FILE}.csv') # load training data
    #df['tau_lep_charge_diff'] = df['total_charge'] * df['taus_charge_0'] # substitute total_charge by tau_lep_charge_diff
    
    df = df.drop(columns=['weight', 'row_number']) # remove auxiliary columns
    
    features = []
    # read features from the csv file
    with open(f'{PATH_FEATURES}{FEATURES_FILE}.csv', 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            features.append(row[0])
    df = df[features]
    
    if torch.cuda.is_available():
        print("CUDA (GPU) is available.")
        device = 'cuda'
    else:
        print("CUDA (GPU) is not available.")
        device = 'cpu'
    
    train_dataset = torch.tensor(df.values, dtype=torch.float32)
    input_size = train_dataset.shape[1]

    # Chose how to scale the data
    scaler = StandardScaler() # MinMaxScaler() 
    train_dataset_norm = scaler.fit_transform(train_dataset)
    train_dataloader = DataLoader(train_dataset_norm, batch_size=gen_params["batch_size"], shuffle=True)

    if APPROACH != 'ddgm': model = VAE(gen_params["latent_size"], device, input_size, conf_dict)
    else: model = DDGM(gen_params["latent_size"], device, input_size, conf_dict)
    
    optimizer = optim.Adam(model.parameters(), lr=gen_params["lr"])

    directory = f'{APPROACH}_{gen_params["num_epochs"]}_epochs_model/'

    if not os.path.exists(f'{PATH_MODEL}{directory}'):
        os.makedirs(f'{PATH_MODEL}{directory}')

    if APPROACH == 'std' or APPROACH == 'std_h' or APPROACH == 'ddgm':
        if TRAIN:
            model.train()
            for epoch in range(gen_params["num_epochs"]):
                progress_bar = tqdm(total=len(train_dataloader))
                for _, x in enumerate(train_dataloader):
                    x = x.view(-1, input_size)
                    optimizer.zero_grad()
                    loss = model(x.float())
                    loss.backward()
                    optimizer.step()
                    progress_bar.set_description(f'EPOCH: {epoch+1}/{gen_params["num_epochs"]} | LOSS: {loss:.7f}')
                    progress_bar.update(1)
                progress_bar.close()     
                elbo_history.append(loss.item())
                if elbo_min > loss.item(): # save the model only if loss is the best so far
                    print("SAVING NEW BEST MODEL")
                    torch.save(model, f'{PATH_MODEL}{directory}{DATA_FILE}.pth')
                    elbo_min = loss.item()
                    plot_loss(elbo_history, None, APPROACH)
                if epoch % 50 == 0: # refresh the loss plot after certain amount of epochs
                    plot_loss(elbo_history, None, APPROACH)
            plot_loss(elbo_history, None, APPROACH)
            data_gen(PATH_DATA, DATA_FILE, PATH_MODEL = f'{PATH_MODEL}{directory}{DATA_FILE}.pth', PATH_JSON=f'{PATH_JSON}hyperparams_gen.json', TYPE=APPROACH, scaler=scaler, reaction=classes[REACTION], dataset=train_dataset_norm, features_list=features)
        else:
            model = torch.load(f'{PATH_MODEL}{directory}{DATA_FILE}.pth')
            print(f'{PATH_MODEL}{directory}{DATA_FILE}.pth')
            data_gen(PATH_DATA, DATA_FILE, PATH_MODEL = f'{PATH_MODEL}{directory}{DATA_FILE}.pth', PATH_JSON=f'{PATH_JSON}hyperparams_gen.json', TYPE=APPROACH, scaler=scaler, reaction=classes[REACTION], dataset=train_dataset_norm, features_list=features)
            
    elif APPROACH == 'sym' or APPROACH == 'sym_h':
        # model = VAE(gen_params["latent_size"], device, input_size, conf_dict)
        
        # optim_encoder_params = list(model.deterministic_encoders.parameters()) + list(model.encoders.parameters()) 
        # optim_decoder_params = list(model.decoders.parameters()) 
        
        # # optimizer1 = optim.Adam(optim_decoder_params, lr=gen_params["lr"])
        # # optimizer2 = optim.Adam(optim_encoder_params, lr=gen_params["lr"])
        
        # optimizer = optim.Adam(model.parameters(), lr=gen_params["lr"])
        
        # directory = f'{APPROACH}_{gen_params["num_epochs"]}_epochs_model/'

        # if not os.path.exists(f'{PATH_MODEL}{directory}'):
        #     os.makedirs(f'{PATH_MODEL}{directory}')
        if TRAIN:
            model.train()
            for epoch in range(gen_params["num_epochs"]):
                progress_bar = tqdm(total=len(train_dataloader))
                for _, x in enumerate(train_dataloader):
                    
                    #! THE FIRST STEP
                    x = x.view(-1, input_size)
                    optimizer.zero_grad()
                    loss1 = model(x.float(), 1)
                    loss1.backward()
                    optimizer.step()
                    
                    #! THE SECOND STEP
                    pz2_gauss = torch.distributions.Normal(torch.zeros((gen_params["batch_size"],gen_params["latent_size"])),
                                                        torch.ones((gen_params["batch_size"],gen_params["latent_size"])))
                    z_2 = pz2_gauss.sample()
                    optimizer.zero_grad()
                    loss2 = model(z_2.float(), 2)
                    loss2.backward()
                    optimizer.step()          
                    progress_bar.set_description(f'EPOCH: {epoch+1}/{gen_params["num_epochs"]} | LOSS 1: {loss1:.7f} | LOSS 2: {loss2:.7f}')
                    progress_bar.update(1)
                progress_bar.close()     
                elbo_history1.append(loss1.item())
                elbo_history2.append(loss2.item())
                if(elbo_min1 > loss1.item() and elbo_min2 > loss2.item()):
                    print("SAVING NEW BEST MODEL")
                    torch.save(model, f'{PATH_MODEL}{directory}{DATA_FILE}.pth') 
                    elbo_min1 = loss1.item()
                    elbo_min2 = loss2.item()
                    plot_loss(elbo_history1, elbo_history2,APPROACH)
                if epoch % 10 == 0:
                    plot_loss(elbo_history1, elbo_history2,APPROACH)
            data_gen(PATH_DATA, DATA_FILE, PATH_MODEL = f'{PATH_MODEL}{directory}{DATA_FILE}.pth', PATH_JSON=f'{PATH_JSON}hyperparams_gen.json', TYPE=APPROACH, scaler=scaler, reaction=classes[REACTION], dataset=train_dataset_norm, features_list=features)
            plot_loss(elbo_history1, elbo_history2, APPROACH)
        else:
            model = torch.load(f'{PATH_MODEL}{directory}{DATA_FILE}.pth')
            print(f'{PATH_MODEL}{directory}{DATA_FILE}.pth')
            data_gen(PATH_DATA, DATA_FILE, PATH_MODEL = f'{PATH_MODEL}{directory}{DATA_FILE}.pth', PATH_JSON=f'{PATH_JSON}hyperparams_gen.json', TYPE=APPROACH, scaler=scaler, reaction=classes[REACTION], dataset=train_dataset_norm, features_list=features)

            

def plot_loss(elbo_history_1, elbo_history_2, TYPE):
    plt.clf()
    epochs = range(1, len(elbo_history_1) + 1)
    plt.plot(epochs, elbo_history_1, label=f'{TYPE} Loss 1')
    if elbo_history_2 is not None:
        plt.plot(epochs, elbo_history_2, label=f'{TYPE} Loss 2')

    plt.xlabel('Epoch')
    plt.ylabel(f'{TYPE} -logp Loss')
    plt.title('Comparison of SYM Losses')
    plt.legend()
    plt.savefig(f'{TYPE}_losses.png')

if __name__ == "__main__":
    main()