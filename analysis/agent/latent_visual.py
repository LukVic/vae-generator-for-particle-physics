import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
#from tsnecuda import TSNE
import umap

import plotly.express as px
import plotly.io as pio

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

import seaborn as sns

import multiprocessing

def dataset_regen(PATH_DATA, DATA_FILE, PATH_MODEL, EPOCHS):
    
    ALGORITHM = 'TSNE'
    directories = [f'std_15000_epochs_model/', f'sym_15000_epochs_model/']
    
    
    # store std and sym together
    data_arrs = []
    encoded_arrs = []
    latent_arrs = []
    decoded_arrs = []
    
    for directory in directories:
    
        print(f'{PATH_MODEL}{directory}{DATA_FILE}_disc_best.pth')
        model = torch.load(f'{PATH_MODEL}{directory}{DATA_FILE}_disc_best.pth')
        df_real = pd.read_csv(f'{PATH_DATA}{DATA_FILE}.csv')
        
        train_dataset = torch.tensor(df_real.values, dtype=torch.float32)
        scaler = StandardScaler()
        train_dataset_norm = torch.tensor(scaler.fit_transform(train_dataset))
        model.eval()
        latent_dimension = 50

        data_arr = []
        encoded_arr= []
        latent_arr = []
        decoded_arr = []

        print("DATASET LOAD DONE")

        for idx, data in enumerate(train_dataset_norm):
            z_means, z_sigmas = model.encoder(data.view(-1, 28).to('cuda').float())
            encoded_arr.extend(z_means.detach().cpu())
            data_arr.append(data)
            
        data_arr = scaler.inverse_transform(np.array(data_arr)).tolist()
        
        with torch.no_grad():
            latent_samples = torch.randn(train_dataset.shape[0], latent_dimension)
            

            decoder_mu_gauss, decoder_sigma_gauss, decoder_bernoulli = model.decoder(latent_samples.to('cuda'))
            px_gauss = torch.distributions.Normal(decoder_mu_gauss, torch.exp(decoder_sigma_gauss))
            decoder_gauss = px_gauss.sample()
            decoder_bernoulli = torch.bernoulli(decoder_bernoulli)
            
            decoder = torch.cat((decoder_gauss, decoder_bernoulli.view(-1,1)), dim=1)
            decoder_denorm = scaler.inverse_transform(decoder.cpu().numpy())
            decoded_arr.extend(decoder_denorm.tolist())
            latent_arr.extend(latent_samples.tolist())
        
        
        data_arrs.append(data_arr)
        encoded_arrs.append(encoded_arr)
        latent_arrs.append(latent_arr)
        decoded_arrs.append(decoded_arr)
    
    
    
    if ALGORITHM == 'TSNE':
        print("RUNNING t-SNE")
        random_state = 42
        tsne_instance = TSNE(n_components=2, perplexity=10, n_jobs=12, random_state=random_state)

        # Fit TSNE transformation arrays separately
        trans_encoded_std = tsne_instance.fit_transform(np.array(encoded_arrs[0]))
        trans_latent_std = tsne_instance.fit_transform(np.array(latent_arrs[0]))
        trans_data_std = tsne_instance.fit_transform(np.array(data_arrs[0]))
        trans_decoded_std = tsne_instance.fit_transform(np.array(decoded_arrs[0]))
        
        trans_encoded_sym = tsne_instance.fit_transform(np.array(encoded_arrs[1]))
        trans_latent_sym = tsne_instance.fit_transform(np.array(latent_arrs[1]))
        trans_data_sym = tsne_instance.fit_transform(np.array(data_arrs[1]))
        trans_decoded_sym = tsne_instance.fit_transform(np.array(decoded_arrs[1]))

        print(trans_encoded_std.shape)
        print(trans_latent_std.shape)
        print(trans_data_std.shape)
        print(trans_decoded_std.shape)
        
        
        print("t-SNE FINISHED")
        
        np.savetxt(f'{PATH_DATA}points_encoded_std.csv',trans_encoded_std, delimiter=',')
        np.savetxt(f'{PATH_DATA}points_latent_std.csv',trans_latent_std, delimiter=',')
        np.savetxt(f'{PATH_DATA}points_data_std.csv',trans_data_std, delimiter=',')
        np.savetxt(f'{PATH_DATA}points_decoded_std.csv',trans_decoded_std, delimiter=',')
        
        np.savetxt(f'{PATH_DATA}points_encoded_sym.csv',trans_encoded_sym, delimiter=',')
        np.savetxt(f'{PATH_DATA}points_latent_sym.csv',trans_latent_sym, delimiter=',')
        np.savetxt(f'{PATH_DATA}points_data_sym.csv',trans_data_sym, delimiter=',')
        np.savetxt(f'{PATH_DATA}points_decoded_sym.csv',trans_decoded_sym, delimiter=',')
        
    else:
        print("RUNNING UMAP")
        trans_encoded = umap.UMAP(n_neighbors=10, n_components=2, n_jobs=12).fit(encoded_arr)
        trans_latent = umap.UMAP(n_neighbors=10, n_components=2, n_jobs=12).fit(latent_arr)
        
        trans_data = umap.UMAP(n_neighbors=10, n_components=2, n_jobs=12).fit(data_arr)
        trans_decoded = umap.UMAP(n_neighbors=10, n_components=2, n_jobs=12).fit(decoded_arr)
        print("UMAP FINISHED")
    
        print(type(trans_encoded))
        print(np.array(trans_encoded.embedding_))
        
        np.savetxt(f'{PATH_DATA}points_encoded_std.csv',trans_encoded.embedding_, delimiter=',')
        np.savetxt(f'{PATH_DATA}points_latent_std.csv',trans_latent.embedding_, delimiter=',')
        np.savetxt(f'{PATH_DATA}points_data_std.csv',trans_data.embedding_, delimiter=',')
        np.savetxt(f'{PATH_DATA}points_decoded_std.csv',trans_decoded.embedding_, delimiter=',')
    



def find_overlap(PATH_DATA, PATH_RESULTS):
    files = ['points_encoded_std.csv', 'points_latent_std.csv', 'points_data_std.csv', 'points_decoded_std.csv']
    points_encoded = np.genfromtxt(f'{PATH_DATA}{files[0]}', delimiter=',')
    points_latent = np.genfromtxt(f'{PATH_DATA}{files[1]}', delimiter=',')
    points_data = np.genfromtxt(f'{PATH_DATA}{files[2]}', delimiter=',')
    points_decoded = np.genfromtxt(f'{PATH_DATA}{files[3]}', delimiter=',')
    
    
    # Initialize sets to keep track of points
    points_encoded_set = set(map(tuple, points_encoded))
    points_latent_set = set(map(tuple, points_latent))
    points_data_set = set(map(tuple, points_data))
    points_decoded_set = set(map(tuple, points_decoded))

    TH = 0.1

    # Initialize lists to store overlapping points
    overlap_encoded = []
    overlap_latent = []
    overlap_data = []
    overlap_decoded = []

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
    umap_encoded_df = pd.DataFrame(data=points_encoded, columns=["t-SNE_1", "t-SNE_2"])
    umap_encoded_df["Source"] = "Encoded Data"
    umap_latent_df = pd.DataFrame(data=points_latent, columns=["t-SNE_1", "t-SNE_2"])
    umap_latent_df["Source"] = "Latent Data"
    # umap_overlap_latent_df = pd.DataFrame(data=overlap_encoded, columns=["t-SNE_1", "t-SNE_2"])
    # umap_overlap_latent_df["Source"] = "Overlapping Data"

    umap_data_df = pd.DataFrame(data=points_data, columns=["t-SNE_1", "t-SNE_2"])
    umap_data_df["Source"] = "Original Data"
    umap_decoded_df = pd.DataFrame(data=points_decoded, columns=["t-SNE_1", "t-SNE_2"])
    umap_decoded_df["Source"] = "Decoded Data"
    # umap_overlap_data_df = pd.DataFrame(data=overlap_data, columns=["t-SNE_1", "t-SNE_2"])
    # umap_overlap_data_df["Source"] = "Overlapping Data"

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
    plt.scatter(umap_encoded_df["t-SNE_1"], umap_encoded_df["t-SNE_2"], c='blue', label='Point Cloud 1 (t-SNE_1)', s = 1)
    plt.title("t-SNE Visualization in 2D (Encoded vs. Latent Data)")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")

    # Plot the second point cloud (t-SNE_2)
    plt.scatter(umap_latent_df["t-SNE_1"], umap_latent_df["t-SNE_2"], c='red', label='Point Cloud 2 (t-SNE_2)', s = 1)

    plt.legend()
    plt.grid(True)
    plt.savefig(f'{PATH_RESULTS}vae_std_latent_tsne.pdf')
    #plt.show()

    

    # Plot the first point cloud (t-SNE_1)
    plt.figure(figsize=(10, 10))
    plt.scatter(umap_decoded_df["t-SNE_1"], umap_decoded_df["t-SNE_2"], c='blue', label='Point Cloud 1 (t-SNE_1)', s = 1)
    plt.title("t-SNE Visualization in 2D (Encoded vs. Latent Data)")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")

    # Plot the second point cloud (t-SNE_2)
    plt.scatter(umap_data_df["t-SNE_1"], umap_data_df["t-SNE_2"], c='red', label='Point Cloud 2 (t-SNE_2)', s = 1)

    plt.legend()
    plt.grid(True)
    plt.savefig(f'{PATH_RESULTS}vae_std_data_tsne.pdf')
    #plt.show()
    
    
    

def main():
    PATH_MODEL = '/home/lucas/Documents/KYR/msc_thesis/vae-generator-for-particle-physics/analysis/models/'
    PATH_DATA = '/home/lucas/Documents/KYR/msc_thesis/vae-generator-for-particle-physics/analysis/data/'
    PATH_RESULTS = '/home/lucas/Documents/KYR/msc_thesis/vae-generator-for-particle-physics/analysis/results/'
    DATA_FILE = 'df_no_zeros'
    EPOCHS = 15000
    
    
    #dataset_regen(PATH_DATA, DATA_FILE, PATH_MODEL, EPOCHS)
    find_overlap(PATH_DATA, PATH_RESULTS)
    
if __name__ == "__main__":
    main()