import numpy as np
import pandas as pd

def con_cat_separation(path):
    features = pd.read_csv(f'{path}features/all_vars_stamp.csv').to_numpy()
    #?print(features) 
    df_big = pd.read_csv(f'{path}data/df_tt.csv')
    df_continuous = pd.DataFrame()
    df_categorical = pd.DataFrame()
    with open(f'{path}features/category_onehot.txt', 'w') as file:
        for feature in features:
            col_name = feature[0]
            stamp = feature[1]
            if stamp == 'con':
                df_continuous[f'{col_name}'] = df_big[f'{col_name}']
            else :
                #print(f"Distinctive values: {df_big[f'{col_name}'].unique().size}")
                for iCat in range(df_big[f'{col_name}'].unique().size):
                    file.write(f'{col_name}_{iCat}' + " ")
                df_categorical[f'{col_name}'] = df_big[f'{col_name}'].astype(int)

    #?print(df_continuous)
    df_continuous.to_csv(path + 'data/df_continuous.csv', index=False)
    df_categorical.to_csv(path + 'data/df_categorical.csv', index=False)
    

def main():    
    PATH = '/home/lucas/Documents/KYR/msc_thesis/vae-generator-for-particle-physics/analysis/'
    con_cat_separation(PATH)

if __name__ == "__main__":
    main()