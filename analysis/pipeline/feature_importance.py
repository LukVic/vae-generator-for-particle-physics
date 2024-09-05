import pandas as pd

from sklearn.model_selection import train_test_split
from mlp_classifier import mlp_classifier
from dataloader import load_config, load_features
from gbdt_classifier import train_model

from aux_functions import *

import torch
from torch.autograd import grad
import torch.nn.functional as F

def feature_importance():
    PATH_JSON = f'../config/' # config path
    gen_params = load_config(PATH_JSON)['classify']['general'] # general parameters for the classification part 
    
    PATH_RESULT = f'../results/{gen_params["sig_mass_label"]}/'
    PATH_DATA = '../data/common/'
    PATH_LOGGING = '../logging/'
    FILE_DATA = 'df_all_pres_strict'
    
    df_data = pd.read_pickle(f'{PATH_DATA}{FILE_DATA}.pkl')
    df_data = df_data[(df_data['sig_mass'] == 0) | (df_data['sig_mass'] == mass_to_num(gen_params['sig_mass_label']))]
    y = df_data['y']
    X = df_data.drop(columns=['y'])
    
    
    # data train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=X['class'], random_state=42)
    helper_features = gen_params['helper_features'] # features helping to analyse the data
    X_train = X_train.drop(columns=list(set(helper_features)-set(['y'])))
    X_test = X_test.drop(columns=list(set(helper_features)-set(['y'])))
    
    if gen_params['deep'][0] == True:
        # MLP
        y_pred_train, y_pred_proba_train, y_pred_test, y_pred_proba_test, accuracy_test, params  = mlp_classifier(X_train, X_test, y_train, y_test, frac_sim=100, frac_gen=0, weights=None) 
        y_pred_proba_train = y_pred_proba_train.numpy()
        y_pred_proba_test = y_pred_proba_test.numpy()
    else:    
        # GBDT
        model = train_model(X_train, y_train, 'XGB')  
        y_pred_train = model.predict(X_train)
        y_pred_proba_train = model.predict_proba(X_train)
        y_pred_test = model.predict(X_test)
        y_pred_proba_test = model.predict_proba(X_test)
    
    





