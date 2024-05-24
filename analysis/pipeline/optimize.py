import optuna 
from optuna import Trial
from datetime import date
import xgboost as xgb
from sklearn.pipeline import Pipeline
from aux_functions import *



def optimize(X_train, y_train, X_test, y_test, weight):
    
    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)
    
    print("Following masses under study.")
    print(str(date.today()))
    study = optuna.create_study(study_name='XGB_Z_LOG: '+ '2024-05-19'+"32",
        direction='maximize', storage='sqlite:///db.sqlite3',load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5))
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_test, y_test, weight), n_trials=1)
    print("Following masses study is finished.")

    print('Best trial: score {}, params {}'.format(study.best_trial.value, study.best_trial.params))
    trials_df = study.trials_dataframe()
    #print(trials_df)
    fig = optuna.visualization.plot_optimization_history(study)
    # fig.show()
    fig=optuna.visualization.plot_intermediate_values(study)
    # fig.show()
    fig = optuna.visualization.plot_param_importances(study)
    # fig.show()
    fig =optuna.visualization.plot_slice(study)
    # fig.show()
    fig = optuna.visualization.plot_parallel_coordinate(study)
    # fig.show()


def objective(trial, X_train, y_train, X_test, y_test, weight):    

    d_train = xgb.DMatrix(X_train, label=y_train)
    d_test = xgb.DMatrix(X_test, label=y_test)
    watchlist = [d_train, d_test]


    parameters_xgb = {
        # "objective": "multi:softprob",
        # "eval_metric": "auc",
        'min_child_weight': trial.suggest_float('min_child_weight', 0.005, 0.05, step=0.001),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0005, 0.005, step = 0.0001),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0001, 10.0, log = True),
        'gamma': trial.suggest_float('gamma', 0.00005, 0.0005, step=0.00001),
        #'num_class': 2,
        # 'tree_method': 'gpu_hist',
        'learning_rate': trial.suggest_float('learning_rate',0.01, 0.1, step = 0.01),
        'n_estimators': trial.suggest_int('n_estimators', 2, 4000, step = 100),
        'colsample_bytree': trial.suggest_float('colsample_bytree',0.2, 1.0, step = 0.1),
        'subsample': trial.suggest_float('subsample',0.2, 1.0, step=0.1),
        #'max_depth': trial.suggest_categorical('max_depth', [6,7,8,9,10,11,12]),
        # 'scale_pos_weight': trial.suggest_categorical('scale_pos_weight', [6, 7, 8, 9, 10, 11, 12]),
        'random_state': 42,
        # 'booster': 'gbtree'
    }

    #clf = XGBClassifier(**parameters_xgb)
    #pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "validation-auc")
    clf = xgb.XGBClassifier()
    clf.set_params(**parameters_xgb)
    
    print('\t|-> training XGBoostClassifier')
    pipe = Pipeline([
        #  ('standard_scaler', sc), 
    #  ('pca', PCA()), 
        ('clf', clf)
    ])

    model = pipe.fit(X_train, y_train)
    y_pred_proba_test = model.predict_proba(X_test)
    
    x_values, y_S, y_B, y_signif_true, y_signif_simp, y_signif, y_signif_imp,  best_significance_threshold, y_BB = calculate_significance_all_thresholds_new_method(y_pred_proba_test, y_test, weight, 0, 5)
    score = threshold_value(x_values, [y_signif_true], ['max'])
    
    print(score)
    
    return score
    
    
def threshold_value(x_values, y_values, optimums, force_threshold_value=None):
    """ Plots graphs for threshold characteristics.
    @param x_values: Values for x-axis (list)
    @param y_values: Values for y-axis (list of lists)
    @param optimums: Whether to search max or min values in y_values
    @param force_threshold_value: Value of forced vertical line value (if None, vertical line is computed with usage of
    optimums parameter)
    @param optimums: Whether to search max or min values in y_values
    @return: the highest significance value
    """
    for values, optimum in zip(y_values, optimums):

        if force_threshold_value is None:
            best_index = 0
            if optimum == 'max':
                best_score = 0
            else:
                best_score = max(values)
            for i, v in enumerate(values):
                if optimum == 'max':
                    if best_score < v:
                        best_score = v
                        best_index = i
                else:
                    if best_score > v:
                        best_score = v
                        best_index = i
        else:
            best_index = int(force_threshold_value * 100)
            best_score = values[best_index]
        
    return best_score