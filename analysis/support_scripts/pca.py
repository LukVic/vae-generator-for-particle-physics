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
    # checking shape
    print('Original Dataframe shape :',X.shape)

    # Mean
    X_mean = X.mean()
    # Standard deviation
    X_std = X.std()
    # Standardization
    Z = (X - X_mean) / X_std
    
    # covariance
    c = Z.cov()
    
    # Plot the covariance matrix
    sns.heatmap(c)
    plt.show()
    
    eigenvalues, eigenvectors = np.linalg.eig(c)
    #print('Eigen values:\n', eigenvalues)
    
    explained_var = np.cumsum(eigenvalues) / np.sum(eigenvalues)
    print(f"Explained variance: {explained_var}")
    
    n_components = np.argmax(explained_var >= 0.50) + 1
    print(f"Number of components: {n_components}")
    
    # PCA component or unit matrix
    u = eigenvectors[:,:n_components]
    pca_component = pd.DataFrame(u)
    
    # plotting heatmap
    plt.figure(figsize =(5, 7))
    sns.heatmap(pca_component)
    plt.title('PCA Component')
    plt.show()

if __name__ == "__main__":
    main()