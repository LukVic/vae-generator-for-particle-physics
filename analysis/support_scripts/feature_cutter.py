import numpy as np
import pandas as pd

def feature_cutter(path):
    features = pd.read_csv(f'{path}features/pt_sum_gen.csv', header=None).to_numpy()
    print(features)
    df_big = pd.read_csv(f'{path}data/df_no_zeros_disc_15000_15000_sym.csv')
    df_ten = pd.DataFrame()
    for feature in features:
        df_ten[f'{feature[0]}'] = df_big[f'{feature[0]}']
    print(df_ten)
    df_ten.to_csv(path + 'data/pt_sum_gen_sym.csv', index=False)
    
def main():    
    PATH = '/home/lucas/Documents/KYR/msc_thesis/vae-generator-for-particle-physics/analysis/'
    feature_cutter(PATH)

if __name__ == "__main__":
    main()