import pandas as pd
import numpy as np


def classification_preprocess():
    classes = {'tth' : 0, 'ttw': 1, 'ttz' : 2, 'tt' : 3}
    
    df_all = pd.DataFrame()
    
    
    for cl in classes:
        PATH_DATA = f'/home/lucas/Documents/KYR/msc_thesis/vae-generator-for-particle-physics/analysis/data/{cl}/'
        DATA_FILE = f'df_{cl}_full_vec'
        
        df_class = pd.read_csv(f'{PATH_DATA}{DATA_FILE}.csv')
        df_class['sig_mass'] = 0
        df_class['y'] = classes[cl]
        
        df_all = pd.concat([df_all, df_class])
    
    
    print(df_all.shape)

    PATH_DATA = '/home/lucas/Documents/KYR/msc_thesis/vae-generator-for-particle-physics/analysis/data/common/'
    FILE_DATA = 'df_all_full_vec'

    df_all.to_pickle(f'{PATH_DATA}{FILE_DATA}.pkl')

def main():
    classification_preprocess()


if __name__ == "__main__":
    main()