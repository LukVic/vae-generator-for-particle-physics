import random
import xgboost as xgb
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


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