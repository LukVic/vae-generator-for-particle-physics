import sys
sys.path.append("/home/lucas/Documents/KYR/msc_thesis/vae-generator-for-particle-physics/analysis/support_scripts")
sys.path.append("/home/lucas/Documents/KYR/msc_thesis/vae-generator-for-particle-physics/analysis/agent")
sys.path.append("/home/lucas/Documents/KYR/msc_thesis/vae-generator-for-particle-physics/analysis/optimizer")
import os

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader

import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer, RobustScaler, PowerTransformer
from tqdm.auto import tqdm

import matplotlib.pyplot as plt

from sample import data_gen
from dataloader import load_config, load_features

from gm_helpers import infer_feature_type
from priors import StandardPrior, MoGPrior, FlowPrior

from adopt import ADOPT

def generate():
    PATH_JSON = f'../config/' # config path
    # load the general parameters
    conf_dict = load_config(PATH_JSON)
    gen_params = conf_dict['generate']['general']
    
    #classes = {'tbh_800': 0, 'tth' : 1, 'ttw': 1, 'ttz' : 1, 'tt' : 1} # multinomimal classification dict
    
    
    TRAIN = gen_params["train"] # train if True, load existing model if False
    #'tbh_all' 'tth' 'ttw' 'ttz' 'tt'
    REACTION = gen_params["reaction"] # 'tbh_800_new' # signal or background
    
    classes = {'tbh_250_new': 0,
            'tbh_800_new': 0,
            'tbh_3000_new': 0,
            'tbh_300': 0,
            'tbh_800': 0,
            'tbh_1500': 0,
            'tbh_2000': 0,
            'bkg_all': 1} # binary classification dict
    
    PATH_DATA = f'../data/{REACTION}_input/' # training data path
    PATH_MODEL = f'../models/production/{REACTION}_input/' # model path
    PATH_FEATURES = f'../features/' # feature directory path

    DATA_FILE = f'df_{REACTION}_pres_strict' # data file based on demanded preselection
    FEATURES_FILE = 'most_important_gbdt_10_tbh_800_new' ##f'features_top_10' # 'df_phi', 'df_no_zeros', 'df_8', 'df_pt'  feature file
    
        # which model to train
    APPROACH = gen_params["approach"] # 'vae_std', 'vae_sym', 'lvae_std', 'lvae_sym', 'ddgm', "gan_std", "wgan_gp"
    
    if APPROACH == 'vae_std' or APPROACH == 'lvae_std' or APPROACH == 'ddgm':
        loss_history = []
        loss_min = np.inf
        if APPROACH == 'vae_std': from architecture_std import VAE
        elif APPROACH == 'lvae_std': from architecture_std_h import VAE
        elif APPROACH == 'ddgm': from architecture_ddgm import DDGM
        
    elif APPROACH == 'vae_sym' or APPROACH == 'lvae_sym' or APPROACH == 'gan_std' or APPROACH == 'wgan_gp':
        loss_history1 = []
        loss_history2 = []
        loss_min1 = np.inf
        loss_min2 = np.inf
        if APPROACH == 'vae_sym': from architecture_sym import VAE
        elif APPROACH == 'lvae_sym': from architecture_sym_h import VAE
        elif APPROACH == 'gan_std': from architecture_gan import GAN
        elif APPROACH == 'wgan_gp': from architecture_gan import WGAN_GP
        
    df = pd.read_csv(f'{PATH_DATA}{DATA_FILE}.csv') # load training data
    
    #df['tau_lep_charge_diff'] = df['total_charge'] * df['taus_charge_0'] # substitute total_charge by tau_lep_charge_diff

    df = df.drop(columns=['weight', 'row_number']) # remove auxiliary columns
    features = load_features(PATH_FEATURES, FEATURES_FILE)
    df = df[features]
    print(df.shape)
    df = df.drop_duplicates(keep='first')
    print(df.shape)
    
    feature_type_dict, df, df_one_hot = infer_feature_type(df,PATH_FEATURES+FEATURES_FILE)
    
    print(f'real_param: {feature_type_dict["real_data"]}')
    print(f'binary_param: {feature_type_dict["binary_data"]}')
    print(f'categorical_param: {feature_type_dict["categorical_one_hot"]}')

    if torch.cuda.is_available():
        print("CUDA (GPU) is available.")
        device = 'cuda'
    else:
        print("CUDA (GPU) is not available.")
        device = 'cpu'
    


    # Chose how to scale the data
    #scaler = MinMaxScaler() 
    #scaler = StandardScaler()
    scaler = QuantileTransformer()
    #scaler = RobustScaler()
    #scaler = PowerTransformer()
    train_dataset_real_norm = scaler.fit_transform(df_one_hot.iloc[:,feature_type_dict['real_data']].values)
    df_one_hot.iloc[:,feature_type_dict['real_data']] = train_dataset_real_norm

    train_dataset_norm = torch.tensor(df_one_hot.values, dtype=torch.float32)

    input_size = train_dataset_norm.shape[1]
    output_dim = feature_type_dict['max_idx']

    train_dataloader = DataLoader(train_dataset_norm, batch_size=gen_params["batch_size"], shuffle=True)
    #TODO refine model parameters
    #prior = MoGPrior(gen_params["latent_size"],5,device)
    #prior = StandardPrior(gen_params["latent_size"])
    #def __init__(self, nets, nett, num_flows, D=2, device=None):
    
    
        # scale (s) network
    M = 256
    nets = lambda: nn.Sequential(nn.Linear(gen_params["latent_size"]// 2, M), nn.GELU(),
                                nn.Linear(M, M), nn.GELU(),
                                nn.Linear(M, gen_params["latent_size"]// 2), nn.Tanh())

    # translation (t) network
    nett = lambda: nn.Sequential(nn.Linear(gen_params["latent_size"]// 2, M), nn.GELU(),
                                nn.Linear(M, M), nn.GELU(),
                                nn.Linear(M, gen_params["latent_size"]// 2))
    
    prior = FlowPrior(nets,nett,10,D=gen_params["latent_size"],device=device)
    
    if APPROACH == 'gan_std': model = GAN(prior,gen_params["latent_size"], device, input_size, conf_dict, output_dim)
    elif APPROACH == 'wgan_gp': model = WGAN_GP(prior,gen_params["latent_size"], device, input_size, conf_dict, output_dim)
    elif APPROACH == 'ddgm': model = DDGM(prior,gen_params["latent_size"], device, input_size, conf_dict, input_size)
    else: model = VAE(prior,gen_params["latent_size"], device, input_size, conf_dict, output_dim)
    
    optimizer = optim.Adam(model.parameters(), lr=gen_params["lr"])
    #optimizer = ADOPT(model.parameters(), lr=gen_params["lr"])
#     scheduler = optim.lr_scheduler.OneCycleLR(
#     optimizer,
#     max_lr=0.0001,  # Can experiment with this
#     steps_per_epoch=len(train_dataloader),
#     epochs=2000,
#     pct_start=0.1,  # Warm-up for the first 10% of training
#     anneal_strategy='cos',  # Use cosine decay for smooth transitions
#     div_factor=10,  # Sets starting lr = max_lr / 10 = 0.0001
#     final_div_factor=1e5  # More aggressive decay at the end
# )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-7)
    # Create the CosineAnnealingWarmRestarts scheduler
    #scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=1, eta_min=1e-6)
    
    directory = f'{APPROACH}_{gen_params["num_epochs"]}_epochs_model/'

    if not os.path.exists(f'{PATH_MODEL}{directory}'):
        os.makedirs(f'{PATH_MODEL}{directory}')

    if APPROACH == 'gan_std' or APPROACH == 'wgan_gp':
        if TRAIN:
            model.train()
            for epoch in range(gen_params["num_epochs"]):
                progress_bar = tqdm(total=len(train_dataloader))
                for _, x in enumerate(train_dataloader):
                    x = x.view(-1, train_dataset_norm.shape[1])
                    optimizer.zero_grad()
                    loss_discriminator, loss_generator = model(x.float(), feature_type_dict)
                    progress_bar.set_description(f'EPOCH: {epoch+1}/{gen_params["num_epochs"]} | LOSS DIS: {loss_discriminator:.7f} | LOSS GEN: {loss_generator:.7f}')
                    progress_bar.update(1)
                #scheduler.step(loss)
                progress_bar.close()     
                loss_history1.append(loss_discriminator.item())
                loss_history2.append(loss_generator.item())
                #if loss_min1 > loss_discriminator.item() and loss_min2 > loss_generator.item(): # save the model only if loss is the best so far
                if loss_min2 > loss_generator.item() or epoch % 50 == 0: # save the model only if loss is the best so far
                    print("SAVING NEW BEST MODEL SO FAR")
                    torch.save(model, f'{PATH_MODEL}{directory}{DATA_FILE}.pth')
                    torch.save(prior, f'{PATH_MODEL}{directory}{DATA_FILE}_prior.pth')
                    loss_min1 = loss_discriminator.item()
                    loss_min2 = loss_generator.item()
                    plot_loss(loss_history1, loss_history2, APPROACH)
                    if epoch % 50 == 0: # refresh the loss plot after certain amount of epochs
                        plot_loss(loss_history1, loss_history2, APPROACH)
            plot_loss(loss_history1, loss_history2, APPROACH)
            data_gen(PATH_DATA, DATA_FILE, PATH_MODEL = f'{PATH_MODEL}{directory}{DATA_FILE}.pth', PATH_JSON=f'{PATH_JSON}', TYPE=APPROACH, scaler=scaler, reaction=classes[REACTION],dataset_original = df , dataset_one_hot=df_one_hot, feature_type_dict=feature_type_dict, features_list=features, prior=prior)
        else:
            model = torch.load(f'{PATH_MODEL}{directory}{DATA_FILE}.pth')
            prior = torch.load(f'{PATH_MODEL}{directory}{DATA_FILE}_prior.pth')
            print(f'{PATH_MODEL}{directory}{DATA_FILE}.pth')
            print(f'{PATH_MODEL}{directory}{DATA_FILE}_prior.pth')
            data_gen(PATH_DATA, DATA_FILE, PATH_MODEL = f'{PATH_MODEL}{directory}{DATA_FILE}.pth', PATH_JSON=f'{PATH_JSON}', TYPE=APPROACH, scaler=scaler, reaction=classes[REACTION],dataset_original = df, dataset_one_hot=df_one_hot, feature_type_dict=feature_type_dict, features_list=features, prior=prior)
            
    
    if APPROACH == 'vae_std' or APPROACH == 'lvae_std' or APPROACH == 'ddgm':
        if TRAIN:
            model.train()
            for epoch in range(gen_params["num_epochs"]):
                progress_bar = tqdm(total=len(train_dataloader))
                for _, x in enumerate(train_dataloader):
                    x = x.view(-1, train_dataset_norm.shape[1])
                    optimizer.zero_grad()
                    loss = model(x.float(), feature_type_dict)
                    loss.backward()
                    optimizer.step()
                    progress_bar.set_description(f'EPOCH: {epoch+1}/{gen_params["num_epochs"]} | LOSS: {loss:.7f}')
                    progress_bar.update(1)
                #scheduler.step(loss)
                progress_bar.close()     
                loss_history.append(loss.item())
                if loss_min > loss.item(): # save the model only if loss is the best so far
                    print("SAVING NEW BEST MODEL")
                    torch.save(model, f'{PATH_MODEL}{directory}{DATA_FILE}.pth')
                    torch.save(prior, f'{PATH_MODEL}{directory}{DATA_FILE}_prior.pth')
                    loss_min = loss.item()
                    plot_loss(loss_history, None, APPROACH)
                if epoch % 50 == 0: # refresh the loss plot after certain amount of epochs
                    plot_loss(loss_history, None, APPROACH)
            plot_loss(loss_history, None, APPROACH)
            data_gen(PATH_DATA, DATA_FILE, PATH_MODEL = f'{PATH_MODEL}{directory}{DATA_FILE}.pth', PATH_JSON=f'{PATH_JSON}', TYPE=APPROACH, scaler=scaler, reaction=classes[REACTION],dataset_original = df , dataset_one_hot=df_one_hot, feature_type_dict=feature_type_dict, features_list=features, prior=prior)
        else:
            model = torch.load(f'{PATH_MODEL}{directory}{DATA_FILE}.pth')
            prior = torch.load(f'{PATH_MODEL}{directory}{DATA_FILE}_prior.pth')
            print(f'{PATH_MODEL}{directory}{DATA_FILE}.pth')
            print(f'{PATH_MODEL}{directory}{DATA_FILE}_prior.pth')
            data_gen(PATH_DATA, DATA_FILE, PATH_MODEL = f'{PATH_MODEL}{directory}{DATA_FILE}.pth', PATH_JSON=f'{PATH_JSON}', TYPE=APPROACH, scaler=scaler, reaction=classes[REACTION],dataset_original = df, dataset_one_hot=df_one_hot, feature_type_dict=feature_type_dict, features_list=features, prior=prior)
            
    elif APPROACH == 'vae_sym' or APPROACH == 'lvae_sym':
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
                    x = x.view(-1, train_dataset_norm.shape[1])
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
                loss_history1.append(loss1.item())
                loss_history2.append(loss2.item())
                if(loss_min1 > loss1.item() and loss_min2 > loss2.item()):
                    print("SAVING NEW BEST MODEL")
                    torch.save(model, f'{PATH_MODEL}{directory}{DATA_FILE}.pth') 
                    loss_min1 = loss1.item()
                    loss_min2 = loss2.item()
                    plot_loss(loss_history1, loss_history2,APPROACH)
                if epoch % 10 == 0:
                    plot_loss(loss_history1, loss_history2,APPROACH)
            data_gen(PATH_DATA, DATA_FILE, PATH_MODEL = f'{PATH_MODEL}{directory}{DATA_FILE}.pth', PATH_JSON=f'{PATH_JSON}', TYPE=APPROACH, scaler=scaler, reaction=classes[REACTION],dataset_original = df, dataset_one_hot=df_one_hot, feature_type_dict=feature_type_dict, features_list=features)
            plot_loss(loss_history1, loss_history2, APPROACH)
        else:
            model = torch.load(f'{PATH_MODEL}{directory}{DATA_FILE}.pth')
            print(f'{PATH_MODEL}{directory}{DATA_FILE}.pth')
            data_gen(PATH_DATA, DATA_FILE, PATH_MODEL = f'{PATH_MODEL}{directory}{DATA_FILE}.pth', PATH_JSON=f'{PATH_JSON}', TYPE=APPROACH, scaler=scaler, reaction=classes[REACTION],dataset_original = df, dataset_one_hot=df_one_hot, feature_type_dict=feature_type_dict, features_list=features)


class DataFrameDataset(Dataset):
    def __init__(self, dataframe, feature_columns, target_column=None):
        self.dataframe = dataframe
        self.feature_columns = feature_columns
        self.target_column = target_column

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # Get features (input data)
        features = self.dataframe.iloc[idx][self.feature_columns].values
        features = torch.tensor(features, dtype=torch.float32)
        
        # Get target (label)
        if self.target_column:
            target = self.dataframe.iloc[idx][self.target_column]
            target = torch.tensor(target, dtype=torch.float32)
            return features, target
        else:
            return features          

def plot_loss(loss_history_1, loss_history_2, TYPE):
    plt.clf()
    epochs = range(1, len(loss_history_1) + 1)
    plt.plot(epochs, loss_history_1, label=f'{TYPE} Loss 1')
    if loss_history_2 is not None:
        plt.plot(epochs, loss_history_2, label=f'{TYPE} Loss 2')

    plt.xlabel('Epoch')
    plt.ylabel(f'{TYPE} -logp Loss')
    plt.title('Comparison of SYM Losses')
    plt.legend()
    plt.savefig(f'{TYPE}_losses.png')


def main():        
    generate()

if __name__ == "__main__":
    main()