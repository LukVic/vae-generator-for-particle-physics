import os
import numpy as np
import pandas as pd
import random
import logging
import matplotlib.pyplot as plt
import torch


from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, auc, roc_curve, precision_score
import umap

import confmatrix_prettyprint as cm

from aux_functions import *
from mlp_classifier import *
from correlation import *
from optimize import optimize

from ucimlrepo import fetch_ucirepo 

def classify():
    OPTIMIZE = True
    TYPE = 'std'
    DEEP = False
    AUGMENT = True
    LOOSE = False
    ONE_MASS = True
    MASS = 2
    #SPLIT_SEED = [56, 17, 98, 42, 73, 5, 61, 89, 25, 10]
    SPLIT_SEED = [32, 56, 17, 98, 42, 73, 5, 61, 89, 25, 10, 3, 12, 93, 45, 7, 68, 27, 94, 71]
    #SPLIT_SEED = [61, 89, 25, 10, 3, 12, 93, 45, 7, 68, 27, 94, 71]
    #SPLIT_SEED = [32]
    #SPLIT_SEED = [32]
    #SPLIT_SEED = [45, 7, 68, 27, 94, 71]
    #SPLIT_SEED = [27, 94, 71]
    
    #sim_fractions = [0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    sim_fractions = [0.99]
    #sim_fractions = [0.2, 0.4, 0.6, 0.8, 0.99]
    aug_fractions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
    #aug_fractions = [0.2, 0.4, 0.6, 0.8, 0.99]
    PATH_DATA = '/home/lucas/Documents/KYR/msc_thesis/vae-generator-for-particle-physics/analysis/data/common/'
    PATH_LOGGING = '/home/lucas/Documents/KYR/msc_thesis/vae-generator-for-particle-physics/analysis/logging/'
    FILE_DATA_LOOSE = 'df_all_pres_loose_jets_only'
    FILE_DATA_STRICT = 'df_all_pres_strict'
    PATH_MODEL =  f'{PATH_DATA}xgboost_model_trained_pres.pkl'
    log_file_path = f'{PATH_LOGGING}results_final.log'
    
    helper_features = ['sig_mass', 'weight', 'row_number', 'file_number', 'y', 'class']
    features_used = ['taus_pt_0', 'MtLepMet', 'met_met', 'DRll01', 'MLepMet', 'minDeltaR_LJ_0', 'jets_pt_0', 'HT', 'HT_lep', 'total_charge']
    # classes = {
    #     0: 'tbh_all',
    #     1: 'tth',
    #     2: 'ttw',
    #     3: 'ttz',
    #     4: 'tt',
    # }
    classes = {
        0: 'tbh_800',
        1: 'bkg_all',
    }
    logg = False
    model = None
    #for frac in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]:
    #for jdx in range(1):
    for ids, seed in enumerate(SPLIT_SEED):
        if logg:
            with open(log_file_path, 'a') as file:
                print("ACCESSING FILE")
                file.write(f'seed_{seed}_params_ = [\n')
        for frac_sim in sim_fractions:
            if frac_sim != 0.99: aug_fractions = [0.0]
            else: aug_fractions = [0.99]#[0.0, 0.2, 0.4, 0.6, 0.8, 0.99]
            for frac_aug in aug_fractions:
                for deep in [False]:
                    df_data_loose = pd.read_pickle(f'{PATH_DATA}{FILE_DATA_LOOSE}.pkl')
                    df_data_strict = pd.read_pickle(f'{PATH_DATA}{FILE_DATA_STRICT}.pkl')
                    df_data_loose = df_data_loose.sample(frac=1.0, random_state=seed)

                    
                    #df_data_loose = df_data_loose[features_used + helper_features]
                    df_data_strict['tau_lep_charge_diff'] = df_data_strict['total_charge'] * df_data_strict['taus_charge_0']
                    df_data_strict = df_data_strict[features_used + helper_features]
                    
                    
                    if ONE_MASS:
                        df_data_loose = df_data_loose[(df_data_loose['sig_mass'] == 0) | (df_data_loose['sig_mass'] == MASS)]
                        df_data_strict = df_data_strict[(df_data_strict['sig_mass'] == 0) | (df_data_strict['sig_mass'] == MASS)]

                    df_data_strict.loc[df_data_strict['y'] == 0, 'weight'] *= 0.403
                    print('SUM')
                    print(df_data_strict.loc[df_data_strict['y'] == 0, 'weight'].sum())
                    y_strict = df_data_strict['y']
                    X_strict = df_data_strict.drop(columns=['y'])
                    # X_strict = X_strict.drop(columns=['total_charge'])
                    # X_strict = X_strict.drop(columns=['lep_ID_1'])
                    # X_strict = X_strict.drop(columns=['lep_ID_0'])
                    # X_strict = X_strict.drop(columns=['taus_fromPV_0'])
                    
                    # df_data_loose = df_data_loose[df_data_loose['weight'] >= 0]
                    # df_data_loose.loc[df_data_loose['y'] == 0, 'weight'] *= 0.005
                    # y_loose = df_data_loose['y']
                    # X_loose = df_data_loose.drop(columns=['y'])
                    X_train_strict, X_test_strict, y_train_strict, y_test_strict = train_test_split(X_strict, y_strict, test_size=0.01, stratify=X_strict['class'], random_state=seed)
                    #X_train_loose, X_test_loose, y_train_loose, y_test_loose = train_test_split(X_loose, y_loose, test_size=0.1, random_state=42)
                    
                    # all_weight_sum = X_strict['weight'].sum()
                    # test_weight_sum = X_test_strict['weight'].sum()
                    
                    # print(f'SUM OF ALL WEIGHTS: {all_weight_sum}')
                    # print(f'SUM OF ALL WEIGHTS: {test_weight_sum}')
                    
                    df_test = pd.concat([X_test_strict, y_test_strict], axis=1)
                    #df_test = df_test[df_test['weight'] > 0]
                    y_test_strict = df_test['y']
                    X_test_strict = df_test.drop(columns=['y'])
                    
                    # # just one mass included
                    if not ONE_MASS:
                        df_test = pd.concat([X_test_strict, y_test_strict], axis=1)
                        df_test = df_test[(df_test['sig_mass'] == 0) | (df_test['sig_mass'] == MASS)]
                        y_test_strict = df_test['y']
                        X_test_strict = df_test.drop(columns=['y'])
                    
                    df_test = pd.concat([X_test_strict, y_test_strict], axis=1)
                    
                    
                    smaller_set = set(zip(df_test['class'], df_test['row_number'], df_test['file_number']))
                    filtered_df_bigger = df_data_loose[~df_data_loose[['class', 'row_number', 'file_number']].apply(tuple, axis=1).isin(smaller_set)]
                    #print(df_data_loose.shape)
                    df_data_loose = filtered_df_bigger
                    #print(df_data_loose.shape)
                    
                    num_rows_train = X_train_strict[X_train_strict['class'] == 4].shape[0]
                    num_rows_test = X_test_strict[X_test_strict['class'] == 4].shape[0]
                    df_data_loose = df_data_loose[df_data_loose['weight'] >= 0]
                    #df_data_loose.loc[df_data_loose['y'] == 0, 'weight'] *= 0.1/0.11016105860471725
                    
                    filtered_values = df_data_loose.loc[df_data_loose['y'] == 0, 'weight']

                    y_loose = df_data_loose['y']
                    
                    X_loose = df_data_loose.drop(columns=['y'])
                    # # X_loose = X_loose.drop(columns=['total_charge'])
                    # # X_loose = X_loose.drop(columns=['lep_ID_1'])
                    # # X_loose = X_loose.drop(columns=['lep_ID_0'])
                    # # X_loose = X_loose.drop(columns=['taus_fromPV_0'])

                    X_train_loose = X_loose
                    y_train_loose = y_loose
                    
                    if LOOSE:
                        X_train = X_train_loose
                        y_train = y_train_loose
                    else:
                        X_train = X_train_strict
                        y_train = y_train_strict
                    
                    df_train = pd.concat([X_train, y_train], axis=1)
                    
                    df_train = df_train.sample(frac=frac_sim, random_state=seed)
                    y_train = df_train['y']
                    X_train = df_train.drop(columns=['y'])
                    
                    X_test = X_test_strict
                    y_test = y_test_strict
                    
                    X_train = X_train.drop(columns=['sig_mass', 'row_number', 'weight', 'class', 'file_number'])
                    #X_train = X_train.drop(columns=['nJets_OR', 'sumPsbtag', 'lep_ID_0', 'lep_ID_1'])
                    
                    weight = X_test['weight']
                    X_test = X_test.drop(columns=['sig_mass', 'row_number', 'weight', 'class', 'file_number'])
                    #X_test = X_test.drop(columns=['nJets_OR', 'sumPsbtag', 'lep_ID_0', 'lep_ID_1'])
                    
                    # X_test = None
                    # y_test = None
                    
                    #! DATA AUGMENTATION
                    if AUGMENT:
                        
                        events_sig = 10449
                        events_bkg = 27611
                        
                        df_train = pd.concat([X_train, y_train], axis=1)
                        df_augment_train = pd.DataFrame()
                        for cl in ['tbh_800_new','bkg_all']:
                            PATH_GEN_MODEL = f'/home/lucas/Documents/KYR/msc_thesis/vae-generator-for-particle-physics/analysis/data/{cl}_input/'
                            if cl == 'tbh_800_new':
                                df_generated = pd.read_csv(f'{PATH_GEN_MODEL}generated_df_{cl}_pres_strict_E20000_S{events_sig}_{TYPE}.csv')
                            else:
                                df_generated = pd.read_csv(f'{PATH_GEN_MODEL}generated_df_{cl}_pres_strict_E20000_S{events_bkg}_{TYPE}.csv')   
                            
                            df_generated = df_generated.sample(frac=frac_aug, random_state=seed)
                            # with open(f'/home/lucas/Documents/KYR/msc_thesis/vae-generator-for-particle-physics/analysis/data/test_dataset/MiniBooNE_PID.txt', 'r') as file:
                            #     lines = file.readlines()

                            # counts = lines[0].strip().split()
                            # positive_count = int(counts[0])
                            # negative_count = int(counts[1])
                            
                            # data = []
                            # class_labels = []
                            # for idx, line in enumerate(lines[1:]):
                            #     data_floats = [float(x) for x in line.split()]
                            #     data.append(data_floats)
                            #     # Assign class labels based on the counts
                            #     if len(class_labels) < positive_count:
                            #         class_labels.append(1)
                            #     else:
                            #         class_labels.append(0)
                            # column_names = [f'Column{i}' for i in range(1, len(data[0])+1)]
                            # df = pd.DataFrame(data, columns=column_names)  
                            # df['y'] = class_labels
                            # df_augment_train = df
                            df_augment_train = pd.concat([df_augment_train, df_generated])
                        
                        
                        y_augment = df_augment_train['y']
                        X_augment = df_augment_train.drop(columns=['y'])
                        
                        #X_train_augment, X_test_augment, y_train_augment, y_test_augment = train_test_split(X_augment,y_augment, test_size=0.2, stratify=df_augment_train['y'], random_state=seed)
                        
                        # print(X_train_augment.shape)
                        # print(X_test_augment.shape)

                        # df_augment_train = pd.concat([X_train_augment, y_train_augment], axis=1)
                        # df_augment_test = pd.concat([X_test_augment, y_test_augment], axis=1)
                        
                        # y_test_aug = df_augment_test[df_augment_test['y'] == 0]

                        
                        #df_train_all = pd.concat([df_train, df_augment_train])
                        df_train_all = df_augment_train

                        
                        # X_test = X_test_augment
                        # y_test = y_test_augment
                        
                        print(df_train)
                        print(df_augment_train)

                        # #wrap_prdc(X_test_strict, df_train)        
                        wasserstein_dist(df_train[df_train['y'] == 0], df_augment_train[df_augment_train['y'] == 0], features_used, cls='SIG')
                        wasserstein_dist(df_train[df_train['y'] == 1], df_augment_train[df_augment_train['y'] == 1], features_used, cls='BKG')
                        # #chi2_test(X_test_strict, df_augment_train, features_used)
                        # correlation_matrix(df_train[df_train['y'] == 0], df_augment_train[df_augment_train['y'] == 0], features_used, cls='SIG', TYPE=TYPE)
                        # correlation_matrix(df_train[df_train['y'] == 1], df_augment_train[df_augment_train['y'] == 1], features_used, cls='BKG', TYPE=TYPE)
                        exit()
                        
                        X_test_strict =X_test_strict.drop(columns=['sig_mass'])
                        X_test_strict =X_test_strict.drop(columns=['row_number'])
                        X_test_strict =X_test_strict.drop(columns=['weight'])
                        X_test_strict =X_test_strict.drop(columns=['class'])
                        
                        # X_train_strict =X_train_strict.drop(columns=['sig_mass'])
                        # X_train_strict =X_train_strict.drop(columns=['row_number'])
                        # X_train_strict =X_train_strict.drop(columns=['weight'])
                        
                        # X_train_loose =X_train_loose.drop(columns=['sig_mass'])
                        # X_train_loose =X_train_loose.drop(columns=['row_number'])
                        # X_train_loose =X_train_loose.drop(columns=['weight'])
                        
                        
                        #compute_embed(X_train.values, X_augment.values, TYPE,3000)
                        #visualize_embed(TYPE)
                        
                        #exit()
                        # df_augment_train =df_augment_train.drop(columns=['y'])
                        
                        df_train_all = df_train_all.sample(frac=1.0, random_state=seed) 
                        df_train_all = df_train_all.reset_index(drop=True)
                        
                        y_train = df_train_all['y']
                        X_train = df_train_all.drop(columns='y')
                    
                    print(f"SHAPE OF THE TRAINING DATASET: {X_train.shape}")
                    #! MODEL IS NOT TRAINED AGAIN IF EXISTS
                    # if os.path.exists(PATH_MODEL):
                    #     model = joblib.load(f'{PATH_DATA}xgboost_model_trained_pres.pkl')
                    # else:

                    model = None
                    y_pred_train, y_pred_proba_train , y_pred_test, y_pred_proba_test = None, None, None, None
                    
                    accuracy_test = 0
                    params = 2
                    
                    
                    if OPTIMIZE:
                        optimize(X_train, y_train, X_test, y_test, weight)
                        continue
                    
                    
                    if deep:
                        print(X_train.shape) 
                        print(X_test.shape) 
                        y_pred_train, y_pred_proba_train, y_pred_test, y_pred_proba_test, accuracy_test, params  = mlp_classifier(X_train, X_test, y_train, y_test, frac_sim, frac_aug, None)
                        y_pred_proba_train = y_pred_proba_train.numpy()
                        y_pred_proba_test = y_pred_proba_test.numpy()
                    else:
                        model = train_model(X_train, y_train, 'XGB')
                        #plot_feature_importnace(PATH_DATA, model, X_train)
                        
                        y_pred_train = model.predict(X_train)
                        y_pred_proba_train = model.predict_proba(X_train)
                        y_pred_test = model.predict(X_test)
                        y_pred_proba_test = model.predict_proba(X_test)
                        
                    accuracy_train = accuracy_score(y_train, y_pred_train)
                    f1_train = f1_score(y_train, y_pred_train, average='macro')
                    auc_train = roc_auc_score(y_train, y_pred_proba_train[:,0], average='macro')
                    precision_train = precision_score(y_train, y_pred_train)
                    
                    print("Accuracy train:", accuracy_train)
                    print("f1 train:", f1_train)
                    print("AUC train:", auc_train)
                    print("Precision train:", precision_train)
                    
                    accuracy_test = accuracy_score(y_test, y_pred_test)
                    f1_test = f1_score(y_test, y_pred_test, average='macro')
                    auc_test = roc_auc_score(y_test, y_pred_proba_test[:,0], average='macro')
                    precision_test = precision_score(y_test, y_pred_test)
                    
                    print("Accuracy test:", accuracy_test)
                    print("f1 test:", f1_test)
                    print("AUC test:", auc_test)
                    print("Precision test:", precision_test)
                    
                    joblib.dump(model,PATH_MODEL)
                    
                    plot_roc_multiclass('XGB'+'_%s test (20%% of %s)' % ('', 'tbH 800'), y_test, y_pred_proba_test, classes,
                                                    {'Accuracy': accuracy_test, 'F1 Weighted': f1_test, 'ROC AUC Weighted': auc_test}, PATH_DATA)
                    
                    
                    x_values, y_S, y_B, y_signif, y_signif_simp, y_signif_imp, y_signif_true, best_significance_threshold, y_BB = calculate_significance_all_thresholds_new_method(y_pred_proba_test, y_test, weight, 0, 5)
                    

                    maxsignificance = max(y_signif)
                    maxsignificance_simple = max(y_signif_simp)
                    # Signal to threshold
                    bs, bb = plot_threshold(x_values, [y_S, y_B], ['max', 'min'], '',
                                    'Expected events',
                                    ['green', 'sienna'], ['S - tbH classifed as tbH', 'B - background classified as tbH'], savepath=f"{PATH_DATA}sb_to_thresh.pdf",
                                    force_threshold_value=best_significance_threshold)
                    
                    # Significance
                    best_signif_true, best_signif_simp = plot_threshold(x_values, [y_signif, y_signif_simp, y_signif_imp, y_signif_true], ['max', 'max', 'max', 'max'], 
                                    '',
                                    'Significance Approximation', ['darkblue','r','darkred', 'purple'],
                                    [r'Z1',r'Z2',r'Z3',r'Z4',],
                                    savepath=f"{PATH_DATA}significance.pdf")
                    
                    print(f"ADVANCED SIGNIFICANCE FORMULA RESULT: {best_signif_true}")
                    
                    cm.plot_confusion_matrix_from_data(y_test, calculate_class_predictions_basedon_decision_threshold(y_pred_proba_test, best_significance_threshold), weight*5,PATH_DATA, pred_val_axis='col')

                    print(len(y_pred_proba_test[:,0]))
                    plot_ouput(weight = weight, y_true=y_test, y_probs=y_pred_proba_test[:,0], PATH_RESULTS=PATH_DATA)
                    logging.basicConfig(filename=f'{PATH_LOGGING}results_final.log', level=logging.INFO, format='%(message)s')

                    
                    
                    # logging.info(f'FRG: {aug_fraction} | ACC TR: {accuracy_train} | ACC TE: {accuracy_test} | SIG: {best_signif_true}')

                    if logg:
                        with open(log_file_path, 'a') as file:
                            # line_to_write = f'\'DEEP: {deep} | FRS: {frac_sim:.1f} | FRG: {frac_aug:.1f} | PREC TR: {precision_train:.5f} | PREC TE: {precision_test:.5f} | ACC TR: {accuracy_train:.5f} | ACC TE: {accuracy_test:.5f} | SIGT: {best_signif_true:.5f} | SIGS: {best_signif_simp:.5f} | BS: {bs:.5f} | BB: {bb:.5f} | W. RAT: {all_weight_sum/test_weight_sum:.5f} | PARAMS: {params}\',\n'
                            line_to_write = f'\'DEEP: {deep} | FRS: {frac_sim:.1f} | FRG: {frac_aug:.1f} | PREC TR: {precision_train:.5f} | PREC TE: {precision_test:.5f} | ACC TR: {accuracy_train:.5f} | ACC TE: {accuracy_test:.5f} | SIGT: {best_signif_true:.5f} | SIGS: {best_signif_simp:.5f} | BS: {bs:.5f} | BB: {bb:.5f} | PARAMS: {params}\',\n'
                            file.write(line_to_write)
                    print("FINISHED")
            
        with open(log_file_path, 'a') as file:
            file.write(f']\n')

def train_model(X, y, name=None, parameters={}, svd_components=0):
    print('ml_utils.train_model():')

    sc = StandardScaler()

    if name == 'XGB':
        grid = {'max_depth': 4, 'gamma': 0,
        'learning_rate': 0.01, 'n_estimators': 80,
        'min_child_weight': 0.01, 'subsample': 0.5, 'colsample_bytree': 0.9,
        'scale_pos_weight': 7, 'seed': 23,'num_class': 7,
            'tree_method': 'gpu_hist'}

        grid_empty = {
            # 'n_estimators': 2300
            'colsample_bytree': 1.0, 'gamma': 0.00025, 'learning_rate': 0.5, 'max_depth': 6, 'min_child_weight': 0.043,
            'n_estimators': 80, 'reg_alpha': 0.0036000000000000003, 'scale_pos_weight': 10, 'subsample': 0.8,
            'num_class': 5,
            'tree_method': 'gpu_hist',
            'random_state': random.randint(0, 1000)

        }
        
        grid_binary = {
            # 'n_estimators': 2300
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'colsample_bytree': 1.0, 'gamma': 0.00025, 'learning_rate': 0.5, 'max_depth': 6, 'min_child_weight': 0.043,
            'n_estimators': 100, 'reg_alpha': 0.0036000000000000003, 'scale_pos_weight': 10, 'subsample': 0.8,
            'tree_method': 'gpu_hist',
            #'random_state': random.randint(0, 1000)

        }
        
        optuna_grid = {'min_child_weight': 0.036,
        'reg_alpha': 0.0029000000000000002,
        'reg_lambda': 0.0003167589991796362,
        'gamma': 0.0005,
        'learning_rate': 0.03,
        'n_estimators': 1102,
        'colsample_bytree': 0.9
        , 'subsample': 1.0
        }
        
        grid_simple = {
        #    'max_depth': 2,
        #    'subsample': 0.1
        }
        
        grid_test = {
            #'n_estimators': 1000,
            #'subsample': 0.9,
            #'eta': 0.1,
            #'reg_alpha': 0.1,
            #'tree_method': 'gpu_hist'
        }
        
        #params {'min_child_weight': 0.036,
        # 'reg_alpha': 0.0029000000000000002,
        # 'reg_lambda': 0.0003167589991796362,
        # 'gamma': 0.0005,
        # 'learning_rate': 0.03,
        # 'n_estimators': 1102,
        # 'colsample_bytree': 0.9
        # , 'subsample': 1.0}
        
        clf = xgb.XGBClassifier()

        clf.set_params(**optuna_grid)
        #clf.set_params()
        
        # XGBoost clssifier
        print('\t|-> training XGBoostClassifier')
        pipe = Pipeline([
          #  ('standard_scaler', sc), 
        #  ('pca', PCA()), 
            ('clf', clf)
        ])

        pipe.fit(X, y)
        
    if name == 'TAB':
        # Parameters for TabNet
        tabnet_params = {
        'n_steps' : 3,
        'gamma' : 1.3,
        'n_shared' : 1,
        'mask_type' : 'entmax',
        'verbose' : 1,
        'device_name' : 'cuda',
        'optimizer_fn' : torch.optim.Adam,
        'optimizer_params' : dict(lr = 2e-2, weight_decay =3e-6),
        'momentum' : 0.7,
        'scheduler_fn' : torch.optim.lr_scheduler.ReduceLROnPlateau,
        'epsilon' : 1e-16,
        'scheduler_params' : dict(mode="min",
                                patience=9,  
                                min_lr=1e-5,
                                factor=0.5)
        }

        # Initialize TabNetClassifier
        clf = TabNetClassifier(**tabnet_params)

        # TabNet classifier
        print('\t|-> training TabNetClassifier')
        pipe = Pipeline([
            ('standard_scaler', sc),
            ('clf', clf)
        ])

        pipe.fit(X, y)

    return pipe



def main():
    classify()

if __name__ == "__main__":
    main()