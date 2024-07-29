import pandas as pd
import numpy as np

# Iterate through the all reactions
# Add idx = class index as a new column
# The sig_mass column distinguished between signal masses => 0 for background
# The y columns distinguished only between signal and background => [0,1]
# The data is finally concatenated together
def classification_preprocess():
    classes = {'tbh_all': 0, 'tth' : 1, 'ttw': 1, 'ttz' : 1, 'tt' : 1, 'vv': 1, 'other': 1}
    #classes = {'tth' : 0, 'ttw': 1, 'ttz' : 1, 'tt' : 1}
    df_all = pd.DataFrame()
    
    for idx, cl in enumerate(classes):
        df_class = pd.read_csv(f'../data/{cl}_input/df_{cl}_pres_strict.csv')
        #print(len(set(df_class['row_number'])))
        #print(df_class['weight'].sum())
        df_class['class'] = idx
        if cl != 'tbh_all':
            df_class['sig_mass'] = 0
        
        df_class['y'] = classes[cl]
        #if cl == 'tbh_all':
            #print(df_class['sig_mass'])
            #print(set(df_class['taus_charge_0'].values))
            #print((df_class[df_all['sig_mass'] == 0, 'weight'].sum()))
        
        df_all = pd.concat([df_all, df_class])
    df_all.to_pickle('../data/common/df_all_pres_strict.pkl')
    print(len(set(df_all['file_number'])))
def main():
    classification_preprocess()


if __name__ == "__main__":
    main()