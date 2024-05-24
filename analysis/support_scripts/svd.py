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
    
    k = 22 

    svd = TruncatedSVD(n_components=k, algorithm='randomized')

    svd.fit(X)

    U_k = svd.transform(X)
    S_k = svd.singular_values_
    VT_k = svd.components_
    print(S_k)
if __name__ == "__main__":
    main()