import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import logging

from scipy.stats import wasserstein_distance
from scipy.linalg import sqrtm
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

def correlation_matrix(path):
    
    # Configure logging to output to console
    logging.basicConfig(filename='/home/lucas/Documents/KYR/msc_thesis/vae-generator-for-particle-physics/analysis/logging/chi2_test.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    EPOCHS_STD = 6000
    EPOCHS_SYM = 6000
    
    DATASET = 'df_no_zeros'
    FEATURES = 'low_features'
    
    df_original = pd.read_csv(f'{path}data/{DATASET}.csv')
    df_generated_std = pd.read_csv(f'{path}data/{DATASET}_disc_{EPOCHS_STD}_{EPOCHS_SYM}_std.csv')
    df_generated_sym = pd.read_csv(f'{path}data/{DATASET}_disc_{EPOCHS_STD}_{EPOCHS_SYM}_sym.csv')
    
    feature_list = pd.read_csv(f'{path}features/{FEATURES}.csv', header=None).to_numpy()
    
    corr_org = df_original.corr()
    corr_std = df_generated_std.corr()
    corr_sym = df_generated_sym.corr()
    
    # Create a figure and subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 10))

    # Plot the first map on the left subplot
    sns.heatmap(corr_org.round(2), ax=axes[0], annot=True, cmap='coolwarm', annot_kws={"size": 6})
    axes[0].set_title('Correlations - Simulated')

    # # Plot the second map on the right subplot
    # sns.heatmap(corr_std.round(2), ax=axes[1], annot=True, cmap='coolwarm', annot_kws={"size": 6})
    # axes[1].set_title('Correlations - ELBO')
    
    # Plot the second map on the right subplot
    sns.heatmap(corr_sym.round(2), ax=axes[1], annot=True, cmap='coolwarm', annot_kws={"size": 6})
    axes[1].set_title('Correlations - SYM')

    # Adjust layout
    plt.tight_layout()

    # Show the plot
    plt.show()
    
def wasserstein_dist(path):
    # Configure logging to output to console
    logging.basicConfig(filename='/home/lucas/Documents/KYR/msc_thesis/vae-generator-for-particle-physics/analysis/logging/chi2_test.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    EPOCHS_STD = 6000
    EPOCHS_SYM = 6000
    
    DATASET = 'df_no_zeros'
    FEATURES = 'low_features'
    
    data_original = np.array([])
    data_generated_std = np.array([])
    data_generated_sym = np.array([])
    
    
    df_original = pd.read_csv(f'{path}data/{DATASET}.csv')
    df_generated_std = pd.read_csv(f'{path}data/{DATASET}_disc_{EPOCHS_STD}_{EPOCHS_SYM}_std.csv')
    df_generated_sym = pd.read_csv(f'{path}data/{DATASET}_disc_{EPOCHS_STD}_{EPOCHS_SYM}_sym.csv')

    feature_list = pd.read_csv(f'{path}features/{FEATURES}.csv', header=None).to_numpy()
    
    wass_sum_std = 0
    wass_sum_sym = 0
    
    
    logging.info(f'ELBO EPOCHS NUM: {EPOCHS_STD}')
    logging.info(f'SYMM EPOCHS NUM: {EPOCHS_SYM}')
    
    for feature in feature_list:
        data_original = df_original[feature[0]].values
        data_generated_std = df_generated_std[feature[0]].values
        data_generated_sym = df_generated_sym[feature[0]].values
        
        hist_original, bins_original = np.histogram(data_original, bins=20, density=True)
        hist_generated_std, bins_generated_std = np.histogram(data_generated_std, bins=20, density=True)
        hist_generated_sym, bins_generated_sym = np.histogram(data_generated_sym, bins=20, density=True)
    
        wasserstein_dist_std = wasserstein_distance(hist_original, hist_generated_std)
        wasserstein_dist_sym = wasserstein_distance(hist_original, hist_generated_sym)
    
        # # Plot the histograms
        # plt.hist(data1, bins=bins1, alpha=0.5, label='Data 1')
        # plt.hist(data2, bins=bins2, alpha=0.5, label='Data 2')
        # plt.xlabel('Values')
        # plt.ylabel('Frequency')
        # plt.title('Histograms of Data 1 and Data 2')
        # plt.legend()
        # plt.show()

        # Print the computed Wasserstein distance
        print("Wasserstein Distance for ELBO: ", wasserstein_dist_std)
        print("Wasserstein Distance for SYMM: ", wasserstein_dist_sym)
        
        wass_sum_std += wasserstein_dist_std
        wass_sum_sym += wasserstein_dist_sym

    print(f"Wasserstein Distance for ELBO (SUM): {wass_sum_std}")
    print(f"Wasserstein Distance for SYMM (SUM): {wass_sum_sym}")
    

def compute_coverage(path):
    
    # Configure logging to output to console
    logging.basicConfig(filename='/home/lucas/Documents/KYR/msc_thesis/vae-generator-for-particle-physics/analysis/logging/chi2_test.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    EPOCHS_STD = 6000
    EPOCHS_SYM = 6000
    
    k = 5
    metric = 'euclidean'
    
    DATASET = 'df_no_zeros'
    FEATURES = 'low_features'
    
    
    df_original = pd.read_csv(f'{path}data/{DATASET}.csv').values
    df_generated_std = pd.read_csv(f'{path}data/{DATASET}_disc_{EPOCHS_STD}_{EPOCHS_SYM}_std.csv').values
    df_generated_sym = pd.read_csv(f'{path}data/{DATASET}_disc_{EPOCHS_STD}_{EPOCHS_SYM}_sym.csv').values
    
    # Normalize the data
    scaler = StandardScaler()
    original_data_normalized = scaler.fit_transform(df_original)
    generated_data_std_normalized = scaler.transform(df_generated_std)
    generated_data_sym_normalized = scaler.transform(df_generated_sym)
    
    # Fit a k-nearest neighbors model on the original data
    knn_original = NearestNeighbors(n_neighbors=k, metric=metric)
    knn_original.fit(original_data_normalized)

    # Find the nearest neighbors of each generated sample in the original data
    distances_std, _ = knn_original.kneighbors(generated_data_std_normalized)
    distances_sym, _ = knn_original.kneighbors(generated_data_sym_normalized)
    
    # Compute coverage: proportion of generated samples with at least one neighbor in the original data
    coverage_std = np.mean(np.any(distances_std < 3, axis=1))
    coverage_sym = np.mean(np.any(distances_sym < 3, axis=1))
    
    print(f"Coverage for ELBO: {coverage_std}")
    print(f"Coverage for SYM: {coverage_sym}")
    
    
# Compute mmd
def compute_mmd(path):
    
    # Configure logging to output to console
    logging.basicConfig(filename='/home/lucas/Documents/KYR/msc_thesis/vae-generator-for-particle-physics/analysis/logging/chi2_test.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    EPOCHS_STD = 6000
    EPOCHS_SYM = 6000
    
    gamma = 1.0
    
    DATASET = 'df_no_zeros'
    FEATURES = 'low_features'
    
    
    df_original = pd.read_csv(f'{path}data/{DATASET}.csv').values
    df_generated_std = pd.read_csv(f'{path}data/{DATASET}_disc_{EPOCHS_STD}_{EPOCHS_SYM}_std.csv').values
    df_generated_sym = pd.read_csv(f'{path}data/{DATASET}_disc_{EPOCHS_STD}_{EPOCHS_SYM}_sym.csv').values
    
    # Normalize the data
    scaler = StandardScaler()
    original_data_normalized = scaler.fit_transform(df_original)
    generated_data_std_normalized = scaler.transform(df_generated_std)
    generated_data_sym_normalized = scaler.transform(df_generated_sym)
    

    mmd_std = gaussian_mmd(original_data_normalized, generated_data_std_normalized, gamma)
    mmd_sym = gaussian_mmd(original_data_normalized, generated_data_sym_normalized, gamma)

    
    print(f"mmd for ELBO: {mmd_std}")
    print(f"mmd for SYM: {mmd_sym}")

def gaussian_mmd(X, Y, gamma):
    
    
    # Compute squared Euclidean distances between samples
    XX = np.sum(X ** 2, axis=1, keepdims=True)
    YY = np.sum(Y ** 2, axis=1, keepdims=True)
    XY = np.dot(X, Y.T)
    
    # Compute Gaussian kernel evaluations
    XX_sum = np.sum(XX) / (X.shape[0] ** 2)
    YY_sum = np.sum(YY) / (Y.shape[0] ** 2)
    XY_sum = np.sum(XY) / (X.shape[0] * Y.shape[0])

    mmd = np.exp(-gamma * (XX_sum - 2 * XY_sum + YY_sum))
    
    return mmd

    
def main():
    PATH = '/home/lucas/Documents/KYR/msc_thesis/vae-generator-for-particle-physics/analysis/'
    #correlation_matrix(PATH)
    #wasserstein_dist(PATH)
    #compute_coverage(PATH)
    compute_mmd(PATH)
    
if __name__ == "__main__":
    main()