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
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, auc, roc_curve
import umap

import confmatrix_prettyprint as cm

from aux_functions import *

def classify():
    
    AUGMENT = False
    LOOSE = False
    ONE_MASS = True
    MASS = 2
    
    PATH_DATA = '/home/lucas/Documents/KYR/msc_thesis/vae-generator-for-particle-physics/analysis/data/common/'
    FILE_DATA_LOOSE = 'df_all_full_vec_pres_loose_feature_cut'
    FILE_DATA_STRICT = 'df_all_full_vec_pres_feature_cut'
    PATH_MODEL =  f'{PATH_DATA}xgboost_model_trained_pres.pkl'
    
    
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
    
    model = None
    
    df_data_loose = pd.read_pickle(f'{PATH_DATA}{FILE_DATA_LOOSE}.pkl')
    df_data_strict = pd.read_pickle(f'{PATH_DATA}{FILE_DATA_STRICT}.pkl')
    
    print(set(df_data_loose['sig_mass'].values))
    print(set(df_data_strict['sig_mass'].values))
    
    if ONE_MASS:
        df_data_loose = df_data_loose[(df_data_loose['sig_mass'] == 0) | (df_data_loose['sig_mass'] == 2)]
        df_data_strict = df_data_strict[(df_data_strict['sig_mass'] == 0) | (df_data_strict['sig_mass'] == 2)]
    
    print(set(df_data_loose['y'].values))
    print(set(df_data_strict['y'].values))
    
    df_data_strict = df_data_strict[df_data_strict['weight'] >= 0]
    df_data_strict.loc[df_data_strict['y'] == 0, 'weight'] *= 0.005
    y_strict = df_data_strict['y']
    X_strict = df_data_strict.drop(columns=['y'])
    # X_strict = X_strict.drop(columns=['total_charge'])
    # X_strict = X_strict.drop(columns=['lep_ID_1'])
    # X_strict = X_strict.drop(columns=['lep_ID_0'])
    # X_strict = X_strict.drop(columns=['taus_fromPV_0'])
    
    X_train_strict, X_test_strict, y_train_strict, y_test_strict = train_test_split(X_strict, y_strict, test_size=0.2, random_state=42)
    
    #:TODO
    # just one mass included
    if not ONE_MASS:
        df_test = pd.concat([X_test_strict, y_test_strict], axis=1)
        df_test = df_test[(df_test['sig_mass'] == 0) | (df_test['sig_mass'] == MASS)]
        y_test_strict = df_test['y']
        X_test_strict = df_test.drop(columns=['y'])
    
    
    row_numbers_to_exclude = X_test_strict['row_number'].tolist()
    df_data_loose = df_data_loose[~df_data_loose['row_number'].isin(row_numbers_to_exclude)]

    
    df_data_loose = df_data_loose[df_data_loose['weight'] >= 0]
    df_data_loose.loc[df_data_loose['y'] == 0, 'weight'] *= 0.005
    y_loose = df_data_loose['y']
    
    
    X_loose = df_data_loose.drop(columns=['y'])
    # X_loose = X_loose.drop(columns=['total_charge'])
    # X_loose = X_loose.drop(columns=['lep_ID_1'])
    # X_loose = X_loose.drop(columns=['lep_ID_0'])
    # X_loose = X_loose.drop(columns=['taus_fromPV_0'])
    
    X_train_loose = X_loose
    y_train_loose = y_loose
    
    if LOOSE:
        X_train = X_train_loose
        y_train = y_train_loose
    else:
        X_train = X_train_strict
        y_train = y_train_strict
    X_test = X_test_strict
    y_test = y_test_strict
    
    X_train = X_train.drop(columns=['sig_mass'])
    X_train = X_train.drop(columns=['row_number'])
    X_train = X_train.drop(columns=['weight'])
    
    weight = X_test['weight']
    X_test = X_test.drop(columns=['sig_mass'])
    X_test = X_test.drop(columns=['row_number'])
    X_test = X_test.drop(columns=['weight'])

    #! DATA AUGMENTATION
    if AUGMENT:
        df_train = pd.concat([X_train, y_train], axis=1)
        df_augment_train = pd.DataFrame()
        for cl in classes.values():
            PATH_GEN_MODEL = f'/home/lucas/Documents/KYR/msc_thesis/vae-generator-for-particle-physics/analysis/data/{cl}_input/'
            df_generated = pd.read_csv(f'{PATH_GEN_MODEL}generated_df_{cl}_full_vec_pres_loose_feature_cut_E100_S100000_std.csv')
            df_augment_train = pd.concat([df_augment_train, df_generated])
        
        df_train = pd.concat([df_train, df_augment_train])
        
        # X_test_strict =X_test_strict.drop(columns=['sig_mass'])
        # X_test_strict =X_test_strict.drop(columns=['row_number'])
        # X_test_strict =X_test_strict.drop(columns=['weight'])
        
        # X_train_strict =X_train_strict.drop(columns=['sig_mass'])
        # X_train_strict =X_train_strict.drop(columns=['row_number'])
        # X_train_strict =X_train_strict.drop(columns=['weight'])
        
        # X_train_loose =X_train_loose.drop(columns=['sig_mass'])
        # X_train_loose =X_train_loose.drop(columns=['row_number'])
        # X_train_loose =X_train_loose.drop(columns=['weight'])

        # df_augment_train =df_augment_train.drop(columns=['y'])
        
        #compute_embed(X_train_strict.values, df_augment_train.values, X_test_strict.values)
        # visualize_embed(PATH_DATA=PATH_DATA, PATH_RESULTS=PATH_DATA)
        # exit()

        df_train = df_train.sample(frac=1, random_state=42) 
        df_train = df_train.reset_index(drop=True)
        
        y_train = df_train['y']
        X_train = df_train.drop(columns='y')
    
    
    #! MODEL IS NOT TRAINED AGAIN IF EXISTS
    # if os.path.exists(PATH_MODEL):
    #     model = joblib.load(f'{PATH_DATA}xgboost_model_trained_pres.pkl')
    # else:

    model = train_model(X_train, y_train, 'XGB')
    
    plot_feature_importnace(PATH_DATA, model, X_train)
    
    y_pred = model.predict(X_train)
    y_pred_proba = model.predict_proba(X_train)
    
    # Evaluate the model
    accuracy = accuracy_score(y_train, y_pred)
    f1_test = f1_score(y_train, y_pred, average='macro')
    auc_test = roc_auc_score(y_train, y_pred_proba[:,0], average='macro')
    
    print("Accuracy:", accuracy)
    print("f1 test:", f1_test)
    print("AUC test:", auc_test)
    
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    print(y_pred_proba[:,0])
    print(y_pred)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    f1_test = f1_score(y_test, y_pred, average='macro')
    auc_test = roc_auc_score(y_test, y_pred_proba[:,1], average='macro')
    
    print("Accuracy:", accuracy)
    print("f1 test:", f1_test)
    print("AUC test:", auc_test)
    
    joblib.dump(model,PATH_MODEL)
    
    
    plot_roc_multiclass('XGB'+'_%s test (20%% of %s)' % ('', 'tbH 800'), y_test, y_pred_proba, classes,
                                    {'Accuracy': accuracy, 'F1 Weighted': f1_test, 'ROC AUC Weighted': auc_test}, PATH_DATA)
    
    cm.plot_confusion_matrix_from_data(y_test, y_pred, weight,PATH_DATA, pred_val_axis='col')
    
    x_values, y_S, y_B, y_signif, y_signif_simp, y_signif_imp, best_significance_threshold, y_BB = calculate_significance_all_thresholds_new_method(y_pred_proba, y_test, weight, 0, 1/0.2)
    maxsignificance = max(y_signif)
    maxsignificance_simple = max(y_signif_simp)
    
    # Signal to threshold
    plot_threshold(x_values, [y_S, y_B], ['max', 'min'], 'S & B to Threshold characteristics',
                    'Expected events',
                    ['green', 'sienna'], ['S - tbH classifed as tbH', 'B - background classified as tbH'], savepath=f"{PATH_DATA}sb_to_thresh.png",
                    force_threshold_value=best_significance_threshold)

    # Significance
    best_signif_simp = plot_threshold(x_values, [y_signif, y_signif_simp, y_signif_imp], ['max', 'max', 'max'], 
                    'Significance Approximations to Threshold characteristics',
                    'Significance Approximation', ['darkred', 'r', 'purple'],
                    ['S/sqrt(S+B)', 'S/sqrt(B)', 'S/(3/2+sqrt(B))'],
                    savepath=f"{PATH_DATA}significance.png")
    print("FINISHED")

def train_model(X, y, name=None, parameters={}, svd_components=0):
    print('ml_utils.train_model():')

    # Scaler
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
            'n_estimators': 80, 'reg_alpha': 0.0036000000000000003, 'scale_pos_weight': 10, 'subsample': 0.8,
            'tree_method': 'gpu_hist',
            'random_state': random.randint(0, 1000)

        }
        clf = xgb.XGBClassifier()

        clf.set_params(**grid_binary)

        # XGBoost clssifier
        print('\t|-> training XGBoostClassifier')
        pipe = Pipeline([
            ('standard_scaler', sc), 
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
                                patience=9,  # changing scheduler patience to be lower than early stopping patience
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