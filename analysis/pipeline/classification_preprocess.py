import pandas as pd
import numpy as np

def classification_preprocess():
    classes = {'tth' : 0, 'ttw': 1, 'ttz' : 2, 'tt' : 3}
    
    df_all_train = pd.DataFrame()
    df_all_test = pd.DataFrame()
    
    
    for cl in classes:
        PATH_DATA = f'/home/lucas/Documents/KYR/msc_thesis/vae-generator-for-particle-physics/analysis/data/{cl}/'
        DATA_FILE = f'df_{cl}_full_vec'
        
        df_class = pd.read_csv(f'{PATH_DATA}{DATA_FILE}.csv')
        df_class['sig_mass'] = 0
        df_class['y'] = classes[cl]
        
        # Shuffle the DataFrame
        df_class = df_class.sample(frac=1, random_state=42)  # Shuffle with random_state for reproducibility

        # Calculate the number of rows for the 20-80 split
        num_rows = df_class.shape[0]
        split_index = int(0.8 * num_rows)

        # Split the shuffled DataFrame into 20% and 80%
        df_class_train = df_class.iloc[:split_index]
        df_class_test = df_class.iloc[split_index:]

        df_all_train = pd.concat([df_all_train, df_class_train])
        df_all_test = pd.concat([df_all_test, df_class_test])
    
    
    print(df_all_train.shape)
    print(df_all_test.shape)

    PATH_DATA = '/home/lucas/Documents/KYR/msc_thesis/vae-generator-for-particle-physics/analysis/data/common/'
    TRAIN_FILE = f'df_train_full_vec'
    TEST_FILE = f'df_test_full_vec'

    np.savetxt(f'{PATH_DATA}{TRAIN_FILE}.csv',df_all_train, delimiter=',')
    np.savetxt(f'{PATH_DATA}{TEST_FILE}.csv',df_all_test, delimiter=',')

def main():
    classification_preprocess()


if __name__ == "__main__":
    main()