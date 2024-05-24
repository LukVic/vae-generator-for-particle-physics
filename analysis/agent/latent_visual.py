import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
#from tsnecuda import TSNE
from umap import UMAP

import plotly.express as px
import plotly.io as pio

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

import seaborn as sns

import multiprocessing

def dataset_regen(PATH_DATA, DATA_FILE, PATH_MODEL, EPOCHS):
    
    ALGORITHM = 'TSNE'
    directories = [f'std_15000_epochs_model/', f'sym_15000_epochs_model/']
    EVENTS_NUM = 3000
    
    print(directories)
    # store std and sym together
    encoded_arrs = []
    decoded_arrs = []
    
    df_real = pd.read_csv(f'{PATH_DATA}{DATA_FILE}.csv')
    
    train_dataset = torch.tensor(df_real.values, dtype=torch.float32)
    
    shuffled_indices = np.random.permutation(train_dataset.shape[0])
    print(train_dataset.shape[0])
    train_dataset = train_dataset[shuffled_indices, :]
    
    train_dataset = train_dataset[:EVENTS_NUM]
    
    scaler = StandardScaler()
    #scaler = MinMaxScaler()
    train_dataset_norm = torch.tensor(scaler.fit_transform(train_dataset))

    latent_dimension = 50
    
    
    for directory in directories:
    
        print(f'{PATH_MODEL}{directory}{DATA_FILE}_disc_best.pth')
        model = torch.load(f'{PATH_MODEL}{directory}{DATA_FILE}_disc_best.pth')
        model.eval()

        data_arr = []
        encoded_arr= []
        latent_arr = []
        decoded_arr = []

        print("DATASET LOAD DONE")

        for data in train_dataset_norm:
            z_means, z_sigmas = model.encoder(data.view(-1, 28).to('cuda').float())
            qz = torch.distributions.Normal(z_means, torch.exp(z_sigmas))
            encoder = qz.sample()
            #encoded_arr.extend(z_means.detach().cpu())
            encoded_arr.extend(encoder.detach().cpu())
            data_arr.append(data)
            
        data_arr = np.array(train_dataset).tolist() 
        #data_arr = scaler.inverse_transform(np.array(data_arr)).tolist()
        
        with torch.no_grad():
            latent_samples = torch.randn(train_dataset.shape[0], latent_dimension)
            decoder_mu_gauss, decoder_sigma_gauss, decoder_bernoulli = model.decoder(latent_samples.to('cuda'))
            px_gauss = torch.distributions.Normal(decoder_mu_gauss, torch.exp(decoder_sigma_gauss))
            decoder_gauss = px_gauss.sample()
            #decoder_gauss = decoder_mu_gauss
            decoder_bernoulli = torch.bernoulli(decoder_bernoulli)
            decoder_bernoulli[decoder_bernoulli == 0] = -1
            decoder = torch.cat((decoder_gauss, decoder_bernoulli.view(-1,1)), dim=1)
            decoder_denorm = scaler.inverse_transform(decoder.cpu().numpy())
            decoder_denorm = decoder_denorm
            decoded_arr.extend(decoder_denorm.tolist())
            latent_arr.extend(latent_samples.tolist())
        
        encoded_arrs.append(encoded_arr)
        decoded_arrs.append(decoded_arr)
        
    #TODO
    # mash together prior, encoded_std, encoded_sym
    # mash torether data, decoded std, decoded sym
    # exit()
    latent_list = encoded_arrs[0] + encoded_arrs[1] + latent_arr
    data_list = decoded_arrs[0] + decoded_arrs[1] + data_arr
    #data_list = data_arr
    list_length = len(encoded_arrs[0])
    
    if ALGORITHM == 'TSNE':
        print("RUNNING t-SNE")
        random_state = 42
        tsne_instance = TSNE(n_components=2, perplexity=30, n_jobs=12, random_state=random_state)
        

        # trans_encoded_std = tsne_instance.fit_transform(np.array(encoded_arrs[0]))
        # trans_prior_std = tsne_instance.fit_transform(np.array(prior_arrs[0]))
        # trans_data_std = tsne_instance.fit_transform(np.array(data_arrs[0]))
        # trans_decoded_std = tsne_instance.fit_transform(np.array(decoded_arrs[0]))
        
        # trans_encoded_sym = tsne_instance.fit_transform(np.array(encoded_arrs[1]))
        # trans_prior_sym = tsne_instance.fit_transform(np.array(prior_arrs[1]))
        # trans_data_sym = tsne_instance.fit_transform(np.array(data_arrs[1]))
        # trans_decoded_sym = tsne_instance.fit_transform(np.array(decoded_arrs[1]))

        # print(trans_encoded_std.shape)
        # print(trans_prior_std.shape)
        # print(trans_data_std.shape)
        # print(trans_decoded_std.shape)
        
        
        # print("t-SNE FINISHED")
        
        # np.savetxt(f'{PATH_DATA}points_encoded_std.csv',trans_encoded_std, delimiter=',')
        # np.savetxt(f'{PATH_DATA}points_prior_std.csv',trans_prior_std, delimiter=',')
        # np.savetxt(f'{PATH_DATA}points_data_std.csv',trans_data_std, delimiter=',')
        # np.savetxt(f'{PATH_DATA}points_decoded_std.csv',trans_decoded_std, delimiter=',')
        
        # np.savetxt(f'{PATH_DATA}points_encoded_sym.csv',trans_encoded_sym, delimiter=',')
        # np.savetxt(f'{PATH_DATA}points_prior_sym.csv',trans_prior_sym, delimiter=',')
        # np.savetxt(f'{PATH_DATA}points_data_sym.csv',trans_data_sym, delimiter=',')
        # np.savetxt(f'{PATH_DATA}points_decoded_sym.csv',trans_decoded_sym, delimiter=',')

        trans_latent = tsne_instance.fit_transform(np.array(latent_list))
        trans_feature = tsne_instance.fit_transform(np.array(data_list))
        
        trans_encoded_std = trans_latent[0:list_length]
        trans_encoded_sym = trans_latent[list_length:2*list_length]
        trans_prior = trans_latent[2*list_length:]
        
        trans_decoded_std = trans_feature[0:list_length]
        trans_decoded_sym = trans_feature[list_length:2*list_length]
        trans_data = trans_feature[2*list_length:]
        #trans_data = trans_feature
        
        np.savetxt(f'{PATH_DATA}points_encoded_std.csv',trans_encoded_std, delimiter=',')
        np.savetxt(f'{PATH_DATA}points_encoded_sym.csv',trans_encoded_sym, delimiter=',')
        np.savetxt(f'{PATH_DATA}points_prior.csv',trans_prior, delimiter=',')
        
        np.savetxt(f'{PATH_DATA}points_decoded_std.csv',trans_decoded_std, delimiter=',')
        np.savetxt(f'{PATH_DATA}points_decoded_sym.csv',trans_decoded_sym, delimiter=',')
        np.savetxt(f'{PATH_DATA}points_data.csv',trans_data, delimiter=',')
        
        
        
    else:
        print("RUNNING UMAP")
        random_state = 42
        umap_instance = UMAP(n_components=2, n_neighbors=30, min_dist=0.1)

        trans_latent = umap_instance.fit_transform(np.array(latent_list))
        trans_feature = umap_instance.fit_transform(np.array(data_list))

        trans_encoded_std = trans_latent[0:list_length]
        trans_encoded_sym = trans_latent[list_length:2*list_length]
        trans_prior = trans_latent[2*list_length:]

        trans_decoded_std = trans_feature[0:list_length]
        trans_decoded_sym = trans_feature[list_length:2*list_length]
        trans_data = trans_feature[2*list_length:]

        np.savetxt(f'{PATH_DATA}points_encoded_std.csv', trans_encoded_std, delimiter=',')
        np.savetxt(f'{PATH_DATA}points_encoded_sym.csv', trans_encoded_sym, delimiter=',')
        np.savetxt(f'{PATH_DATA}points_prior.csv', trans_prior, delimiter=',')

        np.savetxt(f'{PATH_DATA}points_decoded_std.csv', trans_decoded_std, delimiter=',')
        np.savetxt(f'{PATH_DATA}points_decoded_sym.csv', trans_decoded_sym, delimiter=',')
        np.savetxt(f'{PATH_DATA}points_data.csv', trans_data, delimiter=',')
        print("UMAP FINISHED")
       


def find_overlap(PATH_DATA, PATH_RESULTS):
    #files = ['points_encoded_std.csv', 'points_prior_std.csv', 'points_data_std.csv', 'points_decoded_std.csv']
    files = ['points_encoded_std.csv', 'points_encoded_sym.csv', 'points_prior.csv', 'points_decoded_std.csv', 'points_decoded_sym.csv', 'points_data.csv']
    
    
    points_encoded_std = np.genfromtxt(f'{PATH_DATA}{files[0]}', delimiter=',')
    points_encoded_sym = np.genfromtxt(f'{PATH_DATA}{files[1]}', delimiter=',')
    points_prior = np.genfromtxt(f'{PATH_DATA}{files[2]}', delimiter=',')
    points_decoded_std = np.genfromtxt(f'{PATH_DATA}{files[3]}', delimiter=',')
    points_decoded_sym = np.genfromtxt(f'{PATH_DATA}{files[4]}', delimiter=',')
    points_data = np.genfromtxt(f'{PATH_DATA}{files[5]}', delimiter=',')

    tsne_encoded_std_df = pd.DataFrame(data=points_encoded_std, columns=["t-SNE_1", "t-SNE_2"])
    tsne_encoded_std_df["Source"] = "Encoded Data ELBO"
    tsne_encoded_sym_df = pd.DataFrame(data=points_encoded_sym, columns=["t-SNE_1", "t-SNE_2"])
    tsne_encoded_sym_df["Source"] = "Encoded Data SYMM"
    tsne_prior_df = pd.DataFrame(data=points_prior, columns=["t-SNE_1", "t-SNE_2"])
    tsne_prior_df["Source"] = "Prior Data"
    
    tsne_decoded_std_df = pd.DataFrame(data=points_decoded_std, columns=["t-SNE_1", "t-SNE_2"])
    tsne_decoded_std_df["Source"] = "Decoded Data ELBO"
    tsne_decoded_sym_df = pd.DataFrame(data=points_decoded_sym, columns=["t-SNE_1", "t-SNE_2"])
    tsne_decoded_sym_df["Source"] = "Decoded Data SYMM"
    tsne_data_df = pd.DataFrame(data=points_data, columns=["t-SNE_1", "t-SNE_2"])
    tsne_data_df["Source"] = "Simulated Data"
    

    

    plt.figure(figsize=(10, 10))
    plt.scatter(tsne_encoded_sym_df["t-SNE_1"], tsne_encoded_sym_df["t-SNE_2"], c='red', label='Encoded SYM', s=4, alpha=1.0)
    plt.scatter(tsne_encoded_std_df["t-SNE_1"], tsne_encoded_std_df["t-SNE_2"], c='blue', label='Encoded ELBO', s=4, alpha=1.0)
    plt.scatter(tsne_prior_df["t-SNE_1"], tsne_prior_df["t-SNE_2"], c='green', label='Prior', s=4, alpha=1.0)
    plt.title(r"Comparison in the latent space $\mathcal{Z}$", fontsize=36)
    plt.xlabel("t-SNE Component 1", fontsize=28)
    plt.ylabel("t-SNE Component 2", fontsize=28) 
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    legend1 = plt.legend(fontsize=28, scatterpoints=20) 
    legend1.get_frame().set_alpha(0.8)
    plt.grid(True)
    plt.savefig(f'{PATH_RESULTS}vae_latent_tsne_{15000}_30_old.pdf')

    plt.figure(figsize=(10, 10))
    plt.scatter(tsne_decoded_sym_df["t-SNE_1"], tsne_decoded_sym_df["t-SNE_2"], c='red', label='Decoded SYM', s=4, alpha=1.0)
    plt.scatter(tsne_decoded_std_df["t-SNE_1"], tsne_decoded_std_df["t-SNE_2"], c='blue', label='Decoded ELBO', s=4, alpha=1.0)
    plt.scatter(tsne_data_df["t-SNE_1"], tsne_data_df["t-SNE_2"], c='green', label='Simulated data', s=4, alpha=1.0)
    plt.title(r"Comparison in the data space $\mathcal{X}$", fontsize=36)
    plt.xlabel("t-SNE Component 1", fontsize=28)  
    plt.ylabel("t-SNE Component 2", fontsize=28)  
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    legend2 = plt.legend(fontsize=28, scatterpoints=20) 
    legend2.get_frame().set_alpha(0.8)
    plt.grid(True)
    plt.savefig(f'{PATH_RESULTS}vae_data_tsne_{15000}_30_old.pdf')
    
    

def main():
    PATH_MODEL = '/home/lucas/Documents/KYR/msc_thesis/vae-generator-for-particle-physics/analysis/models/'
    PATH_DATA = '/home/lucas/Documents/KYR/msc_thesis/vae-generator-for-particle-physics/analysis/data/'
    PATH_RESULTS = '/home/lucas/Documents/KYR/msc_thesis/vae-generator-for-particle-physics/analysis/results/'
    DATA_FILE = 'df_no_zeros'
    #DATA_FILE = 'df_phi'
    #DATA_FILE = 'df_pt'
    EPOCHS = 1000
    
    dataset_regen(PATH_DATA, DATA_FILE, PATH_MODEL, EPOCHS)
    find_overlap(PATH_DATA, PATH_RESULTS)
    
if __name__ == "__main__":
    main()