from aux_functions import *
from mlp_classifier import *
from correlation import *
import pandas as pd



def cross_validate():
    MASS = 2
    TYPE = 'sym'
    PATH_DATA = '/home/lucas/Documents/KYR/msc_thesis/vae-generator-for-particle-physics/analysis/data/common/'
    PATH_LOGGING = '/home/lucas/Documents/KYR/msc_thesis/vae-generator-for-particle-physics/analysis/logging/'
    FILE_DATA_STRICT = 'df_all_pres_strict'
    SPLIT_SEED = [32, 56, 17, 98, 42, 73, 5, 61, 89, 25, 10, 3, 12, 93, 45, 7, 68, 27, 94, 71]
    sim_fractions = [0.2, 0.4, 0.6, 0.8, 0.99]
    aug_fractions = [0.2, 0.4, 0.6, 0.8, 0.99]
    log_file_path = f'{PATH_LOGGING}data_cross_valid.log'
    
    helper_features = ['sig_mass', 'weight', 'row_number', 'file_number', 'y', 'class']
    features_used = ['taus_pt_0', 'MtLepMet', 'met_met', 'DRll01', 'MLepMet', 'minDeltaR_LJ_0', 'jets_pt_0', 'HT', 'HT_lep', 'total_charge']
    
    classes = {
        0: 'tbh_800',
        1: 'bkg_all',
    }
    
    events_sig = 10449
    events_bkg = 27611
    
    #seed = SPLIT_SEED[0]
    for seed in SPLIT_SEED:
        frac_sim = sim_fractions[-1]
        frac_aug = aug_fractions[-1]
        
        df_data_strict = pd.read_pickle(f'{PATH_DATA}{FILE_DATA_STRICT}.pkl')
        df_data_strict['tau_lep_charge_diff'] = df_data_strict['total_charge'] * df_data_strict['taus_charge_0']
        

        df_data_strict = df_data_strict[features_used + helper_features]
        
        df_data_strict = df_data_strict[(df_data_strict['sig_mass'] == 0) | (df_data_strict['sig_mass'] == MASS)]


        #df_data_strict = df_data_strict[df_data_strict['weight'] >= 0]
        #df_data_strict.loc[df_data_strict['y'] == 0, 'weight'] *= 0.1/0.11016105860471725
        y_strict = df_data_strict['y']
        X_strict = df_data_strict.drop(columns=['y'])
        
        X_train_simul, X_test_simul, y_train_simul, y_test_simul = train_test_split(X_strict, y_strict, test_size=0.2, stratify=X_strict['class'], random_state=seed)
        
        X_train_simul = X_train_simul.drop(columns=['class', 'file_number', 'row_number', 'weight', 'sig_mass'])
        X_test_simul = X_test_simul.drop(columns=['class', 'file_number', 'row_number', 'weight', 'sig_mass'])
        
        df_train_simul = pd.concat([X_train_simul, y_train_simul], axis=1)
        df_test_simul = pd.concat([X_test_simul, y_test_simul], axis=1)
        df_augment = pd.DataFrame()
        for cl in ['tbh_800_new','bkg_all']:
            PATH_GEN_MODEL = f'/home/lucas/Documents/KYR/msc_thesis/vae-generator-for-particle-physics/analysis/data/{cl}_input/'
            if cl == 'tbh_800_new':
                df_generated = pd.read_csv(f'{PATH_GEN_MODEL}generated_df_{cl}_pres_strict_E100000_S{events_sig}_{TYPE}.csv')
            else:
                df_generated = pd.read_csv(f'{PATH_GEN_MODEL}generated_df_{cl}_pres_strict_E100000_S{events_bkg}_{TYPE}.csv')   
            
            df_generated = df_generated.sample(frac=frac_aug, random_state=seed)
            df_augment = pd.concat([df_augment, df_generated])
        
        y_augment = df_augment['y']
        X_augment = df_augment.drop(columns=['y'])
        
        # X_augment, _ , y_augment, _ = train_test_split(X_augment, y_augment, test_size=0.8, random_state=seed)
        
        X_train_augment, X_test_augment, y_train_augment, y_test_augment = train_test_split(X_augment, y_augment, test_size=0.2, random_state=seed)
        df_train_augment = pd.concat([X_train_augment, y_train_augment], axis=1)
        df_test_augment = pd.concat([X_test_augment, y_test_augment], axis=1)
        #df_train_all = pd.concat([df_train, df_augment])
        
        # print(X_train_augment)
        # print(df_data_strict)
        # exit()
        
        #df_train_all = df_train_simul
        df_train_all = df_train_augment
        
        df_test_all = df_test_simul
        #df_test_all = df_test_augment
        
        # wrap_prdc(X_test_strict, df_train)        
        # wasserstein_dist(df_train_augment[df_train_augment['y'] == 0], df_augment[df_augment['y'] == 0], features_used, cls='SIG')
        # wasserstein_dist(df_train_augment[df_train_augment['y'] == 1], df_augment[df_augment['y'] == 1], features_used, cls='BKG')
        # # chi2_test(X_test_strict, df_augment, features_used)
        # correlation_matrix(df_train_augment[df_train_augment['y'] == 0], df_augment[df_augment['y'] == 0], features_used, cls='SIG')
        # correlation_matrix(df_train_augment[df_train_augment['y'] == 1], df_augment[df_augment['y'] == 1], features_used, cls='BKG')
        

        y_train = df_train_all['y']
        X_train = df_train_all.drop(columns=['y'])
        
        y_test = df_test_all['y']
        X_test = df_test_all.drop(columns=['y'])

        y_pred_train, y_pred_proba_train, y_pred_test, y_pred_proba_test, accuracy_test, params  = mlp_classifier(X_train, X_test, y_train, y_test, frac_sim, frac_aug, None)
        y_pred_proba_train = y_pred_proba_train.numpy()
        y_pred_proba_test = y_pred_proba_test.numpy()
        
        accuracy_train = accuracy_score(y_train, y_pred_train)
        accuracy_test = accuracy_score(y_test, y_pred_test)
        
        print("Accuracy train:", accuracy_train)
        print("Accuracy test:", accuracy_test)
        
        with open(log_file_path, 'a') as file:
            line_to_write = f'\'ACCTR : {accuracy_train} | ACCTE : {accuracy_test} | MODEL: {TYPE} | PARAMS: {params}\',\n'
            file.write(line_to_write)
def main():
    cross_validate()

if __name__ == "__main__":
    main()