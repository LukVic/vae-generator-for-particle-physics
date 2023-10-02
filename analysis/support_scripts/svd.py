import torch
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD

def main():
    PATH_RESULTS = '/home/lucas/Documents/KYR/msc_thesis/vae-generator-for-particle-physics/analysis/results/'
    PATH_DATA = '/home/lucas/Documents/KYR/msc_thesis/vae-generator-for-particle-physics/analysis/data/'
    
    X = pd.read_csv(f'{PATH_DATA}df_continuous.csv')
    # U, S, Vh = np.linalg.svd(X)
    # print(S)
    
    # Assuming train_data is your training dataset
    k = 22  # Number of top singular values/vectors to keep

    # Create the TruncatedSVD object
    svd = TruncatedSVD(n_components=k, algorithm='randomized')

    # Fit and transform the dataset
    svd.fit(X)

    # Get the truncated singular values and vectors
    U_k = svd.transform(X)
    S_k = svd.singular_values_
    VT_k = svd.components_
    print(S_k)
if __name__ == "__main__":
    main()