import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import torch
import logging

from scipy.stats import chi2_contingency
from scipy.stats import wasserstein_distance
from scipy.linalg import sqrtm
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from prdc import *

LOAD_EXTERN = False


def correlation_matrix(df_1, df_2, feature_list, path=None, cls='BOTH'):
    
    # Configure logging to output to console
    logging.basicConfig(filename='/home/lucas/Documents/KYR/msc_thesis/vae-generator-for-particle-physics/analysis/logging/chi2_test.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    
    if LOAD_EXTERN:
        df_original, df_generated_std, df_generated_sym, _ = load_data(path=path)
    
        corr_org = df_original.corr(method='pearson')
        corr_std = df_generated_std.corr(method='pearson')
        corr_sym = df_generated_sym.corr(method='pearson')
    
    corr_org = df_1[feature_list].corr(method='pearson')
    corr_std = df_2[feature_list].corr(method='pearson')
    
    # Create a figure and subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 10))

    #Plot the first map on the left subplot
    sns.heatmap(corr_org.round(2), ax=axes[0], annot=True, cmap='coolwarm', annot_kws={"size": 12})
    axes[0].set_title(f'Correlations - Simulated ({cls})')

    # # Plot the second map on the right subplot
    sns.heatmap(corr_std.round(2), ax=axes[1], annot=True, cmap='coolwarm', annot_kws={"size": 12})
    axes[1].set_title(f'Correlations - Standard ELBO generated ({cls})')
    
    # # Plot the second map on the right subplot
    # sns.heatmap(corr_sym.round(2), ax=axes[1], annot=True, cmap='coolwarm', annot_kws={"size": 6})
    # axes[1].set_title('Correlations - SYM')

    # Adjust layout
    plt.tight_layout()

    # Show the plot
    plt.show()
    
def wasserstein_dist(df_1, df_2, feature_list, path=None, cls='both'):
    # Configure logging to output to console
    logging.basicConfig(filename='/home/lucas/Documents/KYR/msc_thesis/vae-generator-for-particle-physics/analysis/logging/chi2_test.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    if LOAD_EXTERN:
        df_original, df_2, df_generated_sym, feature_list = load_data(path=path)
    
    wass_sum = 0
    for feature in feature_list:
        data_1 = df_1[feature].values
        data_2 = df_2[feature].values
        
        hist_1, bins_1 = np.histogram(data_1, bins=100, density=True)
        hist_2, bins_2 = np.histogram(data_2, bins=100, density=True)
    
        wasserstein = wasserstein_distance(hist_1, hist_2)

        print(f"Wasserstein Distance ({cls}): {feature}", wasserstein)
        
        wass_sum += wasserstein

    print(f"Wasserstein Distance ({cls}) (SUM): {wass_sum}")

def chi2_test(df_1, df_2, features, path=None):
    
    smooth = 1e-9
    chi2_sum = 0
    
    for feature in features:
        data_1 = df_1[feature].values
        data_2 = df_2[feature].values
        
        hist_1, bins_1 = np.histogram(data_1, bins=100, density=True)
        hist_2, bins_2 = np.histogram(data_2, bins=100, density=True)
        
        
        observed_frequencies = np.array([hist_1 + smooth, hist_2 + smooth])
        
        chi2_stat, p_val, dof, expected = chi2_contingency(observed_frequencies)

        # print(f"Chi-squared statistic: {feature}", chi2_stat)
        # print(f"Degrees of freedom: {feature}", dof)
        # print(f"P-value: {feature}", p_val)
        print(f"Chi-squared reduced statistic: {feature}", chi2_stat/dof)
        
        chi2_sum += chi2_stat/dof
    
    print(f"Chi2 test (SUM): {chi2_sum}")

def wrap_prdc(df_1, df_2, path=None):
    # Configure logging to output to console
    logging.basicConfig(filename='/home/lucas/Documents/KYR/msc_thesis/vae-generator-for-particle-physics/analysis/logging/chi2_test.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    if LOAD_EXTERN:
        df_original, df_generated_std, df_generated_sym, _ = load_data(path=path)
    
    
    
    result = compute_prdc(df_1, df_2, nearest_k=5)
    print(result)
    

    
    
# Compute mmd
def compute_mmd(path):
    
    # Configure logging to output to console
    logging.basicConfig(filename='/home/lucas/Documents/KYR/msc_thesis/vae-generator-for-particle-physics/analysis/logging/chi2_test.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    if LOAD_EXTERN:
        df_original, df_generated_std, df_generated_sym, _ = load_data(path=path)
    
    # Normalize the data
    scaler = StandardScaler()
    original_data_normalized = scaler.fit_transform(df_original.iloc[0:1000])
    generated_data_std_normalized = scaler.transform(df_generated_std.iloc[0:1000])
    generated_data_sym_normalized = scaler.transform(df_generated_sym.iloc[0:1000])
    
    mmd_std = mmd(torch.tensor(original_data_normalized), torch.tensor(generated_data_std_normalized))
    mmd_sym = mmd(torch.tensor(original_data_normalized), torch.tensor(generated_data_sym_normalized))
    
    print(f"mmd for ELBO: {mmd_std}")
    print(f"mmd for SYM: {mmd_sym}")


def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0]) + int(target.size()[0])
    total = torch.cat([source, target], dim=0)

    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0 - total1) ** 2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)

def mmd(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target, kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY - YX)
    return loss


def compute_fid(path):
    # Configure logging to output to console
    logging.basicConfig(filename='/home/lucas/Documents/KYR/msc_thesis/vae-generator-for-particle-physics/analysis/logging/chi2_test.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    if LOAD_EXTERN:
        df_original, df_generated_std, df_generated_sym, feature_list = load_data(path=path)
    
    # Normalize the data
    scaler = StandardScaler()
    original_data_normalized = scaler.fit_transform(df_original.iloc[0:100])
    generated_data_std_normalized = scaler.transform(df_generated_std.iloc[0:100])
    generated_data_sym_normalized = scaler.transform(df_generated_sym.iloc[0:100])
    
    # calculate mean and covariance statistics
    mu1, sigma1 = original_data_normalized.mean(axis=0), np.cov(original_data_normalized, rowvar=False)
    mu2, sigma2 = generated_data_sym_normalized.mean(axis=0), np.cov(generated_data_sym_normalized, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2)**2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    
    
    print(f"fid for ELBO: {fid}")


def load_data(path):
    EPOCHS_STD = 6000
    EPOCHS_SYM = 6000
    
    DATASET = 'df_no_zeros'
    FEATURES = 'low_features'
    
    df_original = pd.read_csv(f'{path}data/tt/{DATASET}.csv')
    df_generated_std = pd.read_csv(f'{path}data/tt/{DATASET}_disc_{EPOCHS_STD}_{EPOCHS_SYM}_std.csv')
    df_generated_sym = pd.read_csv(f'{path}data/tt/{DATASET}_disc_{EPOCHS_STD}_{EPOCHS_SYM}_sym.csv')

    feature_list = pd.read_csv(f'{path}features/{FEATURES}.csv', header=None).to_numpy()

    return df_original, df_generated_std, df_generated_sym, feature_list

    
def main():
    PATH = '/home/lucas/Documents/KYR/msc_thesis/vae-generator-for-particle-physics/analysis/'
    #correlation_matrix(PATH)
    #wasserstein_dist(PATH)
    #compute_coverage(PATH)
    #wrap_prdc(PATH)
    #compute_mmd(PATH)
    compute_fid(PATH)
    
if __name__ == "__main__":
    main()
    
# MMD alternative implementations
# def gaussian_mmd(X, Y, gamma):
    
    
#     # Compute squared Euclidean distances between samples
#     XX = np.sum(X ** 2, axis=1, keepdims=True)
#     YY = np.sum(Y ** 2, axis=1, keepdims=True)
#     XY = np.dot(X, Y.T)
    
#     # Compute Gaussian kernel evaluations
#     XX_sum = np.sum(XX) / (X.shape[0] ** 2)
#     YY_sum = np.sum(YY) / (Y.shape[0] ** 2)
#     XY_sum = np.sum(XY) / (X.shape[0] * Y.shape[0])

#     mmd = np.exp(-gamma * (XX_sum - 2 * XY_sum + YY_sum))
    
#     return mmd

# def MMD(x, y, kernel):
    
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     """Emprical maximum mean discrepancy. The lower the result
#        the more evidence that distributions are the same.

#     Args:
#         x: first sample, distribution P
#         y: second sample, distribution Q
#         kernel: kernel type such as "multiscale" or "rbf"
#     """
#     xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
#     rx = (xx.diag().unsqueeze(0).expand_as(xx))
#     ry = (yy.diag().unsqueeze(0).expand_as(yy))
    
#     dxx = rx.t() + rx - 2. * xx # Used for A in (1)
#     dyy = ry.t() + ry - 2. * yy # Used for B in (1)
#     dxy = rx.t() + ry - 2. * zz # Used for C in (1)
    
#     XX, YY, XY = (torch.zeros(xx.shape).to(device),
#                   torch.zeros(xx.shape).to(device),
#                   torch.zeros(xx.shape).to(device))
    
#     if kernel == "multiscale":
        
#         bandwidth_range = [0.2, 0.5, 0.9, 1.3]
#         for a in bandwidth_range:
#             XX += a**2 * (a**2 + dxx.to(device))**-1
#             YY += a**2 * (a**2 + dyy.to(device))**-1
#             XY += a**2 * (a**2 + dxy.to(device))**-1
            
#     if kernel == "rbf":
        
#         bandwidth_range = [10, 15, 20, 50]
#         for a in bandwidth_range:
#             XX += torch.exp(-0.5*dxx.to(device)/a)
#             YY += torch.exp(-0.5*dyy.to(device)/a)
#             XY += torch.exp(-0.5*dxy.to(device)/a)

#     return torch.mean(XX + YY - 2. * XY)

# Coverage Outdated
# def compute_coverage(path):
    
#     # Configure logging to output to console
#     logging.basicConfig(filename='/home/lucas/Documents/KYR/msc_thesis/vae-generator-for-particle-physics/analysis/logging/chi2_test.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

#     df_original, df_generated_std, df_generated_sym, _ = load_data(path=path)

#     k = 5
#     metric = 'euclidean'
    
#     # Normalize the data
#     scaler = StandardScaler()
#     original_data_normalized = scaler.fit_transform(df_original)
#     generated_data_std_normalized = scaler.transform(df_generated_std)
#     generated_data_sym_normalized = scaler.transform(df_generated_sym)
    
#     # Fit a k-nearest neighbors model on the original data
#     knn_original = NearestNeighbors(n_neighbors=k, metric=metric)
#     knn_original.fit(original_data_normalized)

#     # Find the nearest neighbors of each generated sample in the original data
#     distances_std, _ = knn_original.kneighbors(generated_data_std_normalized)
#     distances_sym, _ = knn_original.kneighbors(generated_data_sym_normalized)
    
#     # Compute coverage: proportion of generated samples with at least one neighbor in the original data
#     coverage_std = np.mean(np.any(distances_std < 3, axis=1))
#     coverage_sym = np.mean(np.any(distances_sym < 3, axis=1))
    
#     print(f"Coverage for ELBO: {coverage_std}")
#     print(f"Coverage for SYM: {coverage_sym}")