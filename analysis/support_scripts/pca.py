import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    PATH_RESULTS = '/home/lucas/Documents/KYR/msc_thesis/vae-generator-for-particle-physics/analysis/results/'
    PATH_DATA = '/home/lucas/Documents/KYR/msc_thesis/vae-generator-for-particle-physics/analysis/data/'
    
    X = pd.read_csv(f'{PATH_DATA}df_continuous.csv')
    print('Original Dataframe shape :',X.shape)

    X_mean = X.mean()
    X_std = X.std()
    Z = (X - X_mean) / X_std
    
    c = Z.cov()
    
    sns.heatmap(c)
    plt.show()
    
    eigenvalues, eigenvectors = np.linalg.eig(c)
    #print('Eigen values:\n', eigenvalues)
    
    explained_var = np.cumsum(eigenvalues) / np.sum(eigenvalues)
    print(f"Explained variance: {explained_var}")
    
    n_components = np.argmax(explained_var >= 0.50) + 1
    print(f"Number of components: {n_components}")
    
    u = eigenvectors[:,:n_components]
    pca_component = pd.DataFrame(u)
    
    plt.figure(figsize =(5, 7))
    sns.heatmap(pca_component)
    plt.title('PCA Component')
    plt.show()

if __name__ == "__main__":
    main()