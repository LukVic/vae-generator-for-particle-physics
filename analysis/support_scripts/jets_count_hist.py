import ROOT
import pandas as pd
import numpy as np


def jets_count_hist(path):
    data_jets_pt = np.array([])

    
    df_jets_pt = pd.read_csv(f'{path}data/df_jets_all.csv')

    feature_list = pd.read_csv(f'{path}features/jets.csv', header=None).to_numpy()
    print(df_jets_pt)
    non_zero_counts = (df_jets_pt > 0).sum()
    print(non_zero_counts)

    for index, row in df_jets_pt.iterrows():
        if row['jets_pt_14'] != 0:
            print(row['jets_pt_14'])
            print(index)
            
def main():
    PATH = '/home/lucas/Documents/KYR/msc_thesis/vae-generator-for-particle-physics/analysis/'
    jets_count_hist(PATH)

if __name__ == "__main__":
    main()