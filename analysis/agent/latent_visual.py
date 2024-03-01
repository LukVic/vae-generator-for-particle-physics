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
    directories = [f'std_6000_epochs_model/', f'sym_6000_epochs_model/']
    EVENTS_NUM = -1
    
    print(directories)
    # store std and sym together
    encoded_arrs = []
    decoded_arrs = []
    
    df_real = pd.read_csv(f'{PATH_DATA}{DATA_FILE}.csv')
    
    train_dataset = torch.tensor(df_real.values, dtype=torch.float32)
    
    shuffled_indices = np.random.permutation(train_dataset.shape[0])
    print(shuffled_indices)
    train_dataset = train_dataset[shuffled_indices, :]
    
    train_dataset = train_dataset[:EVENTS_NUM]
    
    scaler = StandardScaler()
    #scaler = MinMaxScaler()
    train_dataset_norm = torch.tensor(scaler.fit_transform(train_dataset))

    latent_dimension = 30
    
    
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
            encoded_arr.extend(z_means.detach().cpu())
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
        

        # # Fit TSNE transformation arrays separately
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

        # Fit UMAP transformation arrays separately
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
    
    # points_encoded = np.genfromtxt(f'{PATH_DATA}{files[0]}', delimiter=',')
    # points_prior = np.genfromtxt(f'{PATH_DATA}{files[1]}', delimiter=',')
    # points_data = np.genfromtxt(f'{PATH_DATA}{files[2]}', delimiter=',')
    # points_decoded = np.genfromtxt(f'{PATH_DATA}{files[3]}', delimiter=',')
    
    points_encoded_std = np.genfromtxt(f'{PATH_DATA}{files[0]}', delimiter=',')
    points_encoded_sym = np.genfromtxt(f'{PATH_DATA}{files[1]}', delimiter=',')
    points_prior = np.genfromtxt(f'{PATH_DATA}{files[2]}', delimiter=',')
    points_decoded_std = np.genfromtxt(f'{PATH_DATA}{files[3]}', delimiter=',')
    points_decoded_sym = np.genfromtxt(f'{PATH_DATA}{files[4]}', delimiter=',')
    points_data = np.genfromtxt(f'{PATH_DATA}{files[5]}', delimiter=',')
    
    
    # # Initialize sets to keep track of points
    # points_encoded_set = set(map(tuple, points_encoded))
    # points_latent_set = set(map(tuple, points_prior))
    # points_data_set = set(map(tuple, points_data))
    # points_decoded_set = set(map(tuple, points_decoded))

    # TH = 0.1

    # # Initialize lists to store overlapping points
    # overlap_encoded = []
    # overlap_latent = []
    # overlap_data = []
    # overlap_decoded = []

    # print("COMPUTING THE OVERLAPING")

    # for idx, point1 in enumerate(points_encoded):
    #     if tuple(point1) in points_encoded_set:
    #         points_encoded_set.remove(tuple(point1))

    #     # Calculate the absolute differences between point1 and all points in points_latent
    #     differences = np.abs(points_latent - point1)
    #     #differences = (points_latent - point1)**2

    #     # Calculate the Manhattan distances (sum of absolute differences)
    #     distances = np.sum(differences, axis=1)

    #     # Find the indices of points in points_latent within the threshold
    #     overlapping_indices = np.where(distances < TH)[0]

    #     if len(overlapping_indices) > 0:
    #         for i in overlapping_indices:
    #             if tuple(points_latent[i]) in points_latent_set:
    #                 overlap_encoded.append(point1)
    #                 overlap_latent.append(points_latent[i])
    #                 #points_latent_set.remove(tuple(points_latent[i]))
    #     # print(idx)
        
    # for idx, point1 in enumerate(points_data):
    #     if tuple(point1) in points_data_set:
    #         points_data_set.remove(tuple(point1))

    #     differences = np.abs(points_decoded - point1)

    #     distances = np.sum(differences, axis=1)

    #     overlapping_indices = np.where(distances < TH)[0]

    #     if len(overlapping_indices) > 0:
    #         for i in overlapping_indices:
    #             if tuple(points_decoded[i]) in points_decoded_set:
    #                 overlap_data.append(point1)
    #                 overlap_decoded.append(points_decoded[i])
    #                 #points_latent_set.remove(tuple(points_latent[i]))
    #     # print(idx)  

    # # Convert overlap_encoded and overlap_latent to sets to remove duplicates
    # overlap_encoded_set = set(map(tuple, overlap_encoded))
    # overlap_latent_set = set(map(tuple, overlap_latent))
    # overlap_data_set = set(map(tuple, overlap_data))
    # overlap_decoded_set = set(map(tuple, overlap_decoded))

    # # Convert sets back to NumPy arrays
    # overlap_encoded = np.array(list(overlap_encoded_set))
    # overlap_latent = np.array(list(overlap_latent_set))
    # overlap_data = np.array(list(overlap_data_set))
    # overlap_decoded = np.array(list(overlap_decoded_set))
    
    
    # # Create DataFrames for UMAP results
    # umap_encoded_df = pd.DataFrame(data=points_encoded, columns=["t-SNE_1", "t-SNE_2"])
    # umap_encoded_df["Source"] = "Encoded Data"
    # umap_latent_df = pd.DataFrame(data=points_prior, columns=["t-SNE_1", "t-SNE_2"])
    # umap_latent_df["Source"] = "Latent Data"
    # umap_overlap_latent_df = pd.DataFrame(data=overlap_encoded, columns=["t-SNE_1", "t-SNE_2"])
    # umap_overlap_latent_df["Source"] = "Overlapping Data"

    # umap_data_df = pd.DataFrame(data=points_data, columns=["t-SNE_1", "t-SNE_2"])
    # umap_data_df["Source"] = "Original Data"
    # umap_decoded_df = pd.DataFrame(data=points_decoded, columns=["t-SNE_1", "t-SNE_2"])
    # umap_decoded_df["Source"] = "Decoded Data"
    # umap_overlap_data_df = pd.DataFrame(data=overlap_data, columns=["t-SNE_1", "t-SNE_2"])
    # umap_overlap_data_df["Source"] = "Overlapping Data"

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
    


    # print("OVERLAPING COMPUTED")
    
    # # Combine the DataFrames
    # combined_latent_umap_df = pd.concat([umap_encoded_df, umap_latent_df, umap_overlap_latent_df], axis=0)
    
    # combined_data_umap_df = pd.concat([umap_data_df, umap_decoded_df, umap_overlap_data_df], axis=0)



    # # Create a 2D scatter plot using Plotly Express for the first dataset
    # fig1 = px.scatter(combined_latent_umap_df, x="t-SNE_1", y="t-SNE_2", color="Source",
    #                 title="t-SNE Visualization in 2D (Encoded vs. Latent Data)", opacity=1)

    # fig2 = px.scatter(combined_data_umap_df, x="t-SNE_1", y="t-SNE_2", color="Source",
    #                     title="t-SNE Visualization in 2D (Original vs. Decoded Data)", opacity=1)

    # # Show the modified 2D plot for the first dataset
    # fig1.show()
    # fig2.show()
    

    # Plot for the first dataset
    # Plot the first point cloud (t-SNE_1)
    plt.figure(figsize=(10, 10))
    plt.title("t-SNE Visualization in 2D (Encoded ELBO vs. Encoded SYM vs. Prior)")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")

    # Plot the second point cloud (t-SNE_2)
    plt.scatter(tsne_encoded_sym_df["t-SNE_1"], tsne_encoded_sym_df["t-SNE_2"], c='red', label='Encoded SYM', s = 1, alpha=1.0)
    plt.scatter(tsne_encoded_std_df["t-SNE_1"], tsne_encoded_std_df["t-SNE_2"], c='blue', label='Encoded ELBO', s = 1, alpha=1.0)
    plt.scatter(tsne_prior_df["t-SNE_1"], tsne_prior_df["t-SNE_2"], c='green', label='Prior', s = 1, alpha=1.0)

    plt.legend()
    plt.grid(True)
    plt.savefig(f'{PATH_RESULTS}vae_latent_tsne_{-1}_30_new_pt.pdf')
    #plt.show()

    

    # Plot the first point cloud (t-SNE_1)
    plt.figure(figsize=(10, 10))
    plt.scatter(tsne_decoded_sym_df["t-SNE_1"], tsne_decoded_sym_df["t-SNE_2"], c='red', label='Decoded SYM', s = 1, alpha=1.0)
    plt.scatter(tsne_decoded_std_df["t-SNE_1"], tsne_decoded_std_df["t-SNE_2"], c='blue', label='Decoded ELBO', s = 1, alpha=1.0)
    plt.scatter(tsne_data_df["t-SNE_1"], tsne_data_df["t-SNE_2"], c='green', label='Simulated data', s = 1, alpha = 1.0)
    plt.title("t-SNE Visualization in 2D (Decoded ELBO vs. Decoded SYM vs. Simulated Data)")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")

    # Plot the second point cloud (t-SNE_2)
    
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{PATH_RESULTS}vae_data_tsne_{-1}_30_new_pt.pdf')
    #plt.show()
    
    
    

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