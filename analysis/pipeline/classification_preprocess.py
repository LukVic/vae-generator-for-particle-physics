import pandas as pd
import numpy as np


def classification_preprocess():
    classes = {'tbh_all': 0, 'tth' : 1, 'ttw': 1, 'ttz' : 1, 'tt' : 1}
    df_all = pd.DataFrame()
    
    for idx, cl in enumerate(classes):
        PATH_DATA = f'/home/lucas/Documents/KYR/msc_thesis/vae-generator-for-particle-physics/analysis/data/{cl}_input/'
        DATA_FILE = f'df_{cl}_pres_loose_lep_charge_only'
        
        df_class = pd.read_csv(f'{PATH_DATA}{DATA_FILE}.csv')
        print(df_class.shape)
        print(len(set(df_class['row_number'])))
        df_class['class'] = idx
        if cl != 'tbh_all':
            df_class['sig_mass'] = 0
        
        df_class['y'] = classes[cl]
        
        df_all = pd.concat([df_all, df_class])
    
    PATH_DATA = '/home/lucas/Documents/KYR/msc_thesis/vae-generator-for-particle-physics/analysis/data/common/'
    FILE_DATA = 'df_all_pres_loose_lep_charge_only'

    df_all.to_pickle(f'{PATH_DATA}{FILE_DATA}.pkl')

def main():
    classification_preprocess()


if __name__ == "__main__":
    main()