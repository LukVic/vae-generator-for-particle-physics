import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
#from sklearn.manifold import TSNE
from tsnecuda import TSNE
import umap

import plotly.express as px
import plotly.io as pio

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

import seaborn as sns

import multiprocessing

def dataset_regen(PATH_DATA, DATA_FILE, PATH_MODEL, EPOCHS):
    
    print(f'{PATH_MODEL}{DATA_FILE}_{EPOCHS}.pth')
    model = torch.load(f'{PATH_MODEL}{DATA_FILE}_disc_{EPOCHS}.pth')
    df_real = pd.read_csv(f'{PATH_DATA}{DATA_FILE}.csv')
    
    train_dataset = torch.tensor(df_real.values, dtype=torch.float32)
    input_size = train_dataset.shape[1]
    scaler = StandardScaler()
    train_dataset_norm = scaler.fit_transform(train_dataset)
    model.eval()
    latent_dimension = 21

    data_arr = []
    
    encoded_arr= []
    latent_arr = []
    decoded_arr = []

    for idx, data in enumerate(train_dataset):
        print(f"EVENT NUMBER: {idx}")
        #if idx == 5000: break
        z_means, z_logvar, p = model.encoder(data.view(-1, 27).to('cuda'))
        
        #print(z_means)
        #print(p)
        combined_tensor = torch.cat((z_means, p), dim=1)
        encoded_arr.extend(combined_tensor.detach().cpu())
        #print(combined_tensor)
        #exit()
        data_arr.append(data)
    print("DATASET LOAD DONE")
    encoded_arr = np.array(encoded_arr)
    data_arr = np.array(data_arr)
    #tsne = TSNE(n_components=2, init="pca", random_state=0)
    
    with torch.no_grad():
        latent_samples = torch.randn(train_dataset.shape[0], latent_dimension)
        x_hats = model.decoder(latent_samples.to('cuda'))
        x_hats_denorm = scaler.inverse_transform(x_hats.cpu().numpy())
        
        decoded_arr.extend(x_hats_denorm.tolist())
        latent_arr.extend(latent_samples.tolist())

    latent_arr = np.array(latent_arr)[:, :10]
    #print(data_arr)
    #exit()
    # Run UMAP for encoded and latent data
    print("RUNNING UMAP")
    trans_encoded = umap.UMAP(n_neighbors=10, n_components=2, n_jobs=12).fit(encoded_arr)
    trans_latent = umap.UMAP(n_neighbors=10, n_components=2, n_jobs=12).fit(latent_arr)
    
    trans_data = umap.UMAP(n_neighbors=10, n_components=2, n_jobs=12).fit(data_arr)
    trans_decoded = umap.UMAP(n_neighbors=10, n_components=2, n_jobs=12).fit(decoded_arr)
    print("UMAP FINISHED")
    
    print(type(trans_encoded))
    print(np.array(trans_encoded.embedding_))
    
    np.savetxt(f'{PATH_DATA}points_encoded.csv',trans_encoded.embedding_, delimiter=',')
    np.savetxt(f'{PATH_DATA}points_latent.csv',trans_latent.embedding_, delimiter=',')
    np.savetxt(f'{PATH_DATA}points_data.csv',trans_data.embedding_, delimiter=',')
    np.savetxt(f'{PATH_DATA}points_decoded.csv',trans_decoded.embedding_, delimiter=',')
    exit()
    

def find_overlap(PATH_DATA):
    files = ['points_encoded.csv', 'points_latent.csv', 'points_data.csv', 'points_decoded.csv']
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

    print("COMPUTING THE OVERLAPING")

    for idx, point1 in enumerate(points_encoded):
        if tuple(point1) in points_encoded_set:
            points_encoded_set.remove(tuple(point1))

        # Calculate the absolute differences between point1 and all points in points_latent
        differences = np.abs(points_latent - point1)
        #differences = (points_latent - point1)**2

        # Calculate the Manhattan distances (sum of absolute differences)
        distances = np.sum(differences, axis=1)

        # Find the indices of points in points_latent within the threshold
        overlapping_indices = np.where(distances < TH)[0]

        if len(overlapping_indices) > 0:
            for i in overlapping_indices:
                if tuple(points_latent[i]) in points_latent_set:
                    overlap_encoded.append(point1)
                    overlap_latent.append(points_latent[i])
                    #points_latent_set.remove(tuple(points_latent[i]))
        print(idx)
        
    for idx, point1 in enumerate(points_data):
        if tuple(point1) in points_data_set:
            points_data_set.remove(tuple(point1))

        differences = np.abs(points_decoded - point1)

        distances = np.sum(differences, axis=1)

        overlapping_indices = np.where(distances < TH)[0]

        if len(overlapping_indices) > 0:
            for i in overlapping_indices:
                if tuple(points_decoded[i]) in points_decoded_set:
                    overlap_data.append(point1)
                    overlap_decoded.append(points_decoded[i])
                    #points_latent_set.remove(tuple(points_latent[i]))
        print(idx)  

    # Convert overlap_encoded and overlap_latent to sets to remove duplicates
    overlap_encoded_set = set(map(tuple, overlap_encoded))
    overlap_latent_set = set(map(tuple, overlap_latent))
    overlap_data_set = set(map(tuple, overlap_data))
    overlap_decoded_set = set(map(tuple, overlap_decoded))

    # Convert sets back to NumPy arrays
    overlap_encoded = np.array(list(overlap_encoded_set))
    overlap_latent = np.array(list(overlap_latent_set))
    overlap_data = np.array(list(overlap_data_set))
    overlap_decoded = np.array(list(overlap_decoded_set))
    print(overlap_encoded.shape)
    print(overlap_data.shape)
    
    
    # Create DataFrames for UMAP results
    umap_encoded_df = pd.DataFrame(data=points_encoded, columns=["UMAP_1", "UMAP_2"])
    umap_encoded_df["Source"] = "Encoded Data"
    umap_latent_df = pd.DataFrame(data=points_latent, columns=["UMAP_1", "UMAP_2"])
    umap_latent_df["Source"] = "Latent Data"
    umap_overlap_latent_df = pd.DataFrame(data=overlap_encoded, columns=["UMAP_1", "UMAP_2"])
    umap_overlap_latent_df["Source"] = "Overlapping Data"

    umap_data_df = pd.DataFrame(data=points_data, columns=["UMAP_1", "UMAP_2"])
    umap_data_df["Source"] = "Original Data"
    umap_decoded_df = pd.DataFrame(data=points_decoded, columns=["UMAP_1", "UMAP_2"])
    umap_decoded_df["Source"] = "Decoded Data"
    umap_overlap_data_df = pd.DataFrame(data=overlap_data, columns=["UMAP_1", "UMAP_2"])
    umap_overlap_data_df["Source"] = "Overlapping Data"

    print("OVERLAPING COMPUTED")
    
    # Combine the DataFrames
    combined_latent_umap_df = pd.concat([umap_encoded_df, umap_latent_df, umap_overlap_latent_df], axis=0)
    
    combined_data_umap_df = pd.concat([umap_data_df, umap_decoded_df, umap_overlap_data_df], axis=0)



    # Create a 2D scatter plot using Plotly Express for the first dataset
    fig1 = px.scatter(combined_latent_umap_df, x="UMAP_1", y="UMAP_2", color="Source",
                    title="UMAP Visualization in 2D (Encoded vs. Latent Data)", opacity=1)

    fig2 = px.scatter(combined_data_umap_df, x="UMAP_1", y="UMAP_2", color="Source",
                        title="UMAP Visualization in 2D (Original vs. Decoded Data)", opacity=1)

    # Show the modified 2D plot for the first dataset
    fig1.show()
    fig2.show()
    

def main():
    PATH_MODEL = '/home/lucas/Documents/KYR/msc_thesis/vae-generator-for-particle-physics/analysis/models/'
    PATH_DATA = '/home/lucas/Documents/KYR/msc_thesis/vae-generator-for-particle-physics/analysis/data/'
    DATA_FILE = 'df_low'
    EPOCHS = 100
    
    
    #dataset_regen(PATH_DATA, DATA_FILE, PATH_MODEL, EPOCHS)
    find_overlap(PATH_DATA)
    
if __name__ == "__main__":
    main()