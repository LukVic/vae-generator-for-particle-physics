import numpy as np
import pandas as pd


def feature_cutter(path):
    
    classes = {'tbh_all'}
    FEATURE_CSV_PATH = 'features/best_5.csv'
    
    for cls in classes:
        
        INPUT_DATSET_CSV_PATH = f'data/{cls}_input/df_{cls}_full_vec_pres_loose.csv'
        OUTPUT_DATASET_CSV_PATH = f'data/{cls}_input/df_{cls}_full_vec_pres_loose_feature_cut.csv'
        
        # FEATURE_CSV_PATH = 'features/pt_sum_gen.csv'
        # INPUT_DATSET_CSV_PATH = 'data/df_no_zeros_disc_6000_6000_sym.csv'
        # OUTPUT_DATASET_CSV_PATH = 'data/pt_sum_gen_sym.csv'
        
        # FEATURE_CSV_PATH = 'features/phi_features.csv'
        # INPUT_DATSET_CSV_PATH = 'data/df_no_zeros.csv'
        # OUTPUT_DATASET_CSV_PATH = 'data/df_phi.csv'
        
        # FEATURE_CSV_PATH = 'features/pt_features.csv'
        # INPUT_DATSET_CSV_PATH = 'data/df_no_zeros.csv'
        # OUTPUT_DATASET_CSV_PATH = 'data/df_pt.csv'
        
        # FEATURE_CSV_PATH = 'features/features_8.csv'
        # INPUT_DATSET_CSV_PATH = 'data/df_no_zeros.csv'
        # OUTPUT_DATASET_CSV_PATH = 'data/df_8.csv'
        
        
        features = pd.read_csv(f'{path}{FEATURE_CSV_PATH}', header=None).to_numpy()
        print(features)
        df_big = pd.read_csv(f'{path}{INPUT_DATSET_CSV_PATH}')
        df_ten = pd.DataFrame()
        for feature in features:
            df_ten[f'{feature[0]}'] = df_big[f'{feature[0]}']
        df_ten.to_csv(f'{path}{OUTPUT_DATASET_CSV_PATH}', index=False)



def main():    
    PATH = '/home/lucas/Documents/KYR/msc_thesis/vae-generator-for-particle-physics/analysis/'
    feature_cutter(PATH)

if __name__ == "__main__":
    main()