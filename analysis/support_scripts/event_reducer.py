import numpy as np
import pandas as pd

def feature_cutter(path):
    df_original = pd.read_csv(f'{path}data/df_low.csv')
    features = np.array(['jets_pt_0', 'jets_pt_1', 'jets_pt_2', 'jets_pt_3'])
    
    mask = (df_original[features] == 0).any(axis=1)
    df_light = df_original[~mask]
    
    df_light.to_csv(path + 'data/df_no_zeros.csv', index=False)
    
def main():    
    PATH = '/home/lucas/Documents/KYR/msc_thesis/vae-generator-for-particle-physics/analysis/'
    feature_cutter(PATH)

if __name__ == "__main__":
    main()