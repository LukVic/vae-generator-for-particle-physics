import os
import numpy as np
import pandas as pd
import random
import logging
import matplotlib.pyplot as plt
import torch

import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, auc, roc_curve, precision_score
import umap


from aux_functions import *
from mlp_classifier import *
from correlation import *

def data_classify_check():
    

    MASS = 2
    seed = 3
    frac_aug = 1.0
    frac_sim = 1.0
    
    
    PATH_DATA = '/home/lucas/Documents/KYR/msc_thesis/vae-generator-for-particle-physics/analysis/data/common/'
    PATH_LOGGING = '/home/lucas/Documents/KYR/msc_thesis/vae-generator-for-particle-physics/analysis/logging/'
    FILE_DATA_STRICT = 'df_all_pres_strict'
    PATH_MODEL =  f'{PATH_DATA}xgboost_model_trained_pres.pkl'
    log_file_path = f'{PATH_LOGGING}data_aug_results.log'
    
    helper_features = ['sig_mass', 'weight', 'row_number', 'file_number', 'y', 'class']
    features_used = ['taus_pt_0', 'MtLepMet', 'met_met', 'DRll01', 'MLepMet', 'minDeltaR_LJ_0', 'jets_pt_0', 'HT', 'HT_lep', 'total_charge']

    classes = {
        0: 'tbh_800',
        1: 'bkg_all',
    }
    logg = False
    model = None

    df_data_strict = pd.read_pickle(f'{PATH_DATA}{FILE_DATA_STRICT}.pkl')  
    df_data_strict = df_data_strict[features_used + helper_features]
    df_data_strict = df_data_strict[(df_data_strict['sig_mass'] == 0) | (df_data_strict['sig_mass'] == MASS)]


    #df_data_strict = df_data_strict[df_data_strict['weight'] >= 0]
    y_strict = df_data_strict['y']
    X_strict = df_data_strict.drop(columns=['y'])


    X_train_strict, X_test_strict, y_train_strict, y_test_strict = train_test_split(X_strict, y_strict, test_size=0.2, stratify=X_strict['class'], random_state=seed)    
    
    
    X_train_strict = X_train_strict.drop(columns=['sig_mass', 'row_number', 'weight', 'class', 'file_number'])
    X_test_strict = X_test_strict.drop(columns=['sig_mass', 'row_number', 'weight', 'class', 'file_number'])
    
    # X_train = X_train_strict
    # y_train = y_train_strict
    
    
    
    X_test = X_test_strict
    y_test = y_test_strict
    

    
    X_train = None
    y_train = None
    #X_test = None
    #y_test = None
    
    events_sig = 10449*50
    events_bkg = 27611*50
    
    df_augment_train = pd.DataFrame()
    df_train = pd.DataFrame()
    for cl in ['tbh_800_new','bkg_all']:
        PATH_GEN_MODEL = f'/home/lucas/Documents/KYR/msc_thesis/vae-generator-for-particle-physics/analysis/data/{cl}_input/'
        if cl == 'tbh_800_new':
            df_generated = pd.read_csv(f'{PATH_GEN_MODEL}generated_df_{cl}_pres_strict_E20000_S{events_sig}_std.csv')
        else:
            df_generated = pd.read_csv(f'{PATH_GEN_MODEL}generated_df_{cl}_pres_strict_E20000_S{events_bkg}_std.csv')   

        df_augment_train = pd.concat([df_augment_train, df_generated])
        
        
    y_augment = df_augment_train['y']
    X_augment = df_augment_train.drop(columns=['y'])
    
    X_train_augment, X_test_augment, y_train_augment, y_test_augment = train_test_split(X_augment,y_augment, test_size=0.02, stratify=df_augment_train['y'], random_state=seed)
    
    print(X_train_augment.shape)
    print(X_test_augment.shape)

    df_augment_train = pd.concat([X_train_augment, y_train_augment], axis=1)
    df_augment_test = pd.concat([X_test_augment, y_test_augment], axis=1)
    
    y_test_aug = df_augment_test[df_augment_test['y'] == 0]


    df_train = pd.concat([X_train_strict, y_train_strict], axis=1)
    #df_train_all = pd.concat([df_train, df_augment_train])
    df_train_all = df_augment_train
    
    #print(df_train_all)
    
    #X_test = X_test_augment
    #y_test = y_test_augment
    
    
    df_train_all = df_train_all.sample(frac=frac_aug, random_state=42) 
    df_train_all = df_train_all.reset_index(drop=True)
    
    y_train = df_train_all['y']
    X_train = df_train_all.drop(columns='y')
    
    print(f"SHAPE OF THE TRAINING DATASET: {X_train.shape}")
    model = None
    y_pred_train, y_pred_proba_train , y_pred_test, y_pred_proba_test = None, None, None, None
    
    accuracy_test = 0
    params = 0
    print(X_train.shape) 
    print(X_test.shape) 
    print(y_test.shape)
    print(y_train.shape)
    y_pred_train, y_pred_proba_train, y_pred_test, y_pred_proba_test, accuracy_test, params  = mlp_classifier(X_train, X_test, y_train, y_test, frac_sim, frac_aug, None)
    y_pred_proba_train = y_pred_proba_train.numpy()
    y_pred_proba_test = y_pred_proba_test.numpy()
        
    accuracy_train = accuracy_score(y_train, y_pred_train)
    precision_train = precision_score(y_train, y_pred_train)
    
    print("Accuracy train:", accuracy_train)
    print("Precision train:", precision_train)
    
    accuracy_test = accuracy_score(y_test, y_pred_test)
    precision_test = precision_score(y_test, y_pred_test)
    
    print("Accuracy test:", accuracy_test)
    print("Precision test:", precision_test)
    
    joblib.dump(model,PATH_MODEL)
    
    plot_ouput(y_true=y_test, y_probs=y_pred_proba_test[:,0], PATH_RESULTS=PATH_DATA)
    
    logging.basicConfig(filename=f'{PATH_LOGGING}data_reduction_signif.log', level=logging.INFO, format='%(message)s')

    print("FINISHED")




def main():
    data_classify_check()

if __name__ == "__main__":
    main()