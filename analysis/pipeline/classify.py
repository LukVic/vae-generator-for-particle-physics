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

import confmatrix_prettyprint as cm


def classify():
    
    AUGMENT = True
    LOOSE = True
    ONE_MASS = False
    MASS = 1
    
    PATH_DATA = '/home/lucas/Documents/KYR/msc_thesis/vae-generator-for-particle-physics/analysis/data/common/'
    FILE_DATA_LOOSE = 'df_all_full_vec_pres_loose_feature_cut'
    FILE_DATA_STRICT = 'df_all_full_vec_pres_feature_cut'
    PATH_MODEL =  f'{PATH_DATA}xgboost_model_trained_pres.pkl'
    
    
    classes = {
        0: 'tbh_all',
        1: 'tth',
        2: 'ttw',
        3: 'ttz',
        4: 'tt',
    }
    # classes = {
    #     0: 'tbh_all',
    #     1: 'bkg_all',
    # }
    
    model = None
    
    df_data_loose = pd.read_pickle(f'{PATH_DATA}{FILE_DATA_LOOSE}.pkl')
    df_data_strict = pd.read_pickle(f'{PATH_DATA}{FILE_DATA_STRICT}.pkl')
    
    if ONE_MASS:
        df_data_loose = df_data_loose[(df_data_loose['sig_mass'] == 0) | (df_data_loose['sig_mass'] == MASS)]
        df_data_strict = df_data_strict[(df_data_strict['sig_mass'] == 0) | (df_data_strict['sig_mass'] == MASS)]
    

    df_data_strict = df_data_strict[df_data_strict['weight'] >= 0]
    df_data_strict.loc[df_data_strict['y'] == 0, 'weight'] *= 0.14
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
    df_data_loose.loc[df_data_loose['y'] == 0, 'weight'] *= 0.14
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
            df_generated = pd.read_csv(f'{PATH_GEN_MODEL}generated_df_{cl}_full_vec_pres_loose_feature_cut_E100_S10000_std.csv')
            print(df_generated)
            print(df_train)
            df_augment_train = pd.concat([df_augment_train, df_generated])
        
        df_train = pd.concat([df_train, df_augment_train])
        df_train = df_train.sample(frac=1, random_state=42) 
        df_train = df_train.reset_index(drop=True)
        
        y_train = df_train['y']
        X_train = df_train.drop(columns='y')
    
    
    #! MODEL IS NOT TRAINED AGAIN IF EXISTS
    # if os.path.exists(PATH_MODEL):
    #     model = joblib.load(f'{PATH_DATA}xgboost_model_trained_pres.pkl')
    # else:

    model = train_model(X_train, y_train, 'XGB')
    
    # Get feature importances
    importances = model.named_steps['clf'].feature_importances_

    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1][:5]  # Select the top 20 indices

    # Plot feature importances
    plt.figure(figsize=(10, 6))
    plt.title("Top 20 Feature Importances")
    plt.bar(range(5), importances[indices], align="center")
    plt.xticks(range(5), X_train.columns[indices], rotation=90)
    plt.xlim([-1, 5])
    plt.tight_layout()
    plt.savefig(f'{PATH_DATA}feature_importance_top20.png')
    print(X_train.columns[indices])
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    f1_test = f1_score(y_test, y_pred, average='macro')
    auc_test = roc_auc_score(y_test, y_pred_proba, average='macro', multi_class='ovr')
    
    print("Accuracy:", accuracy)
    print("f1 test:", f1_test)
    print("AUC test:", auc_test)
    
    joblib.dump(model,PATH_MODEL)
    
    plot_roc_multiclass('XGB'+'_%s test (20%% of %s)' % ('', 'ttH'), y_test, y_pred_proba, classes,
                                    {'Accuracy': accuracy, 'F1 Weighted': f1_test, 'ROC AUC Weighted': auc_test}, PATH_DATA)
    
    cm.plot_confusion_matrix_from_data(y_test, y_pred, weight,PATH_DATA, pred_val_axis='col')
    
    x_values, y_S, y_B, y_signif, y_signif_simp, y_signif_imp, best_significance_threshold = calculate_significance_all_thresholds_new_method(y_pred_proba, y_test, weight, 0, 0.2)
    maxsignificance = max(y_signif)
    maxsignificance_simple = max(y_signif_simp)
    
    # Signal to threshold
    plot_threshold(x_values, [y_S, y_B], ['max', 'min'], 'S & B to Threshold characteristics',
                    'Expected events',
                    ['green', 'sienna'], ['S - LQ classifed as LQ', 'B - background classified as LQ'], savepath=f"{PATH_DATA}sb_to_thresh.png",
                    force_threshold_value=best_significance_threshold)

    # Significance
    best_signif_simp = plot_threshold(x_values, [y_signif, y_signif_simp, y_signif_imp], ['max', 'max', 'max'], 
                    'Significance Approximations to Threshold characteristics',
                    'Significance Approximation', ['darkred', 'r', 'purple'],
                    ['S/sqrt(S+B)', 'S/sqrt(B)', 'S/(3/2+sqrt(B))'],
                    savepath=f"{PATH_DATA}significance.png")


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
        clf = xgb.XGBClassifier()

        clf.set_params(**grid_empty)

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

def plot_roc_multiclass(title, y, y_probas, classes, scores, folder):
    """ Plots ROC curve for multi-class classification.
    @param title: Title of plot
    @param y: True labels
    @param y_probas: Predicted probabilities
    @param classes: Dictionary with classes
    @param scores: Dictionary with scores
    """
    y_bin = label_binarize(y, classes=list(classes.keys()))
    n_classes = y_bin.shape[1]
    fprs = dict()
    tprs = dict()
    aucs = dict()
    for i in range(n_classes):
        fprs[i], tprs[i], _ = roc_curve(y_bin[:, i], y_probas[:, i])
        aucs[i] = auc(fprs[i], tprs[i])
    plt.figure()
    for key in scores:
        plt.plot([0, 0], [0, 0], color='k', linestyle='-', label='{}: {}'.format(key, np.round(scores[key], 2)))
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--', label='Chance')
    for i in range(n_classes):
        plt.plot(fprs[i], tprs[i], label='ROC %d - %s (AUC = %0.2f)' % (i, classes[i], aucs[i]))
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc="lower right")
    # plt.show()
    # plt.draw()
    plt.savefig(folder + 'roc.png')

def calculate_significance_all_thresholds_new_method(predicted_probs_y: np.array, true_y: np.array,
                                                     weights: np.array, other_bgr: float, test_set_scale_factor: float) \
        -> tuple[list[float], list[float], list[float], list[float], list[float], float]:
    """ Obtain characteristics for simplified significance maximization threshold tuning
    @param predicted_probs_y: Predicted probabilities
    @param true_y: Ground truth labels
    @param scale_factors: Weighting factors used for comparison with existing results
    @param N_events: Number of events in real data used for comparison
    @param other_bgr: Number of events in other background in real data used for comparison
    @return: Graph data, best threshold
    """
    x_values = list()
    y_S = list()
    y_B = list()
    y_signif = list()
    y_signif_simp = list()
    y_signif_imp = list()
    max_sig = 0
    best_th = 0
    threshold_start = 0
    logger = logging.getLogger()
    for th in np.round(np.arange(threshold_start, 1, 0.01), 2):
        th = np.round(th, 3)
        x_values.append(th)
        if th % 0.2 == 0:
            logger.info('Threshold: {}'.format(th))
        y_pred = calculate_class_predictions_basedon_decision_threshold(predicted_probs_y, th)
        response = calculate_significance_one_threshold_new_method(true_y, y_pred, weights, other_bgr,
                                                                   test_set_scale_factor)
        
        y_S.append(response['S'])
        y_B.append(response['B'])
        y_signif.append(response['significance'])
        y_signif_simp.append(response['significance_simple'])
        y_signif_imp.append(response['significance_improved'])

        if response['significance_improved'] > max_sig:
            max_sig = response['significance_improved']
            best_th = th
    return x_values, y_S, y_B, y_signif, y_signif_simp, y_signif_imp, best_th

def calculate_class_predictions_basedon_decision_threshold(predicted_probs_y: np.array, threshold: float):
    """ Outputs an array with predicted classes based on predicted class probabilities and decision threshold
    If the predicted probability for class 0 is greater than threshold, the predicted class is 0,
    otherwise it is max P(y|x) among the remaining classes
    @param predicted_probs_y: Predicted probabilities
    @param threshold: Decision threshold
    @return: Predicted classes

    """

    length = np.shape(predicted_probs_y)[0]
    width = np.shape(predicted_probs_y)[1]
    y_pred = list()
    for i in range(length):
        if predicted_probs_y[i, 0] >= threshold:
            y_pred.append(0)
        else:
            predicted_class = 0
            max_p = 0
            for j in range(1, width):
                if max_p < predicted_probs_y[i, j]:
                    max_p = predicted_probs_y[i, j]
                    predicted_class = j
            y_pred.append(predicted_class)
    return y_pred

def calculate_significance_one_threshold_new_method(true_y: np.array, predicted_y: np.array, weights: np.array,
                                                    other_background=0, \
                                                    test_set_scale_factor: float = 1):
    """ Approximate significance score for prediction.
        Used to compare with known results for given weights, number of events, and background compensation constant.
    @param true_y: Ground truth labels
    @param predicted_y: Predicted probabilities
    @param weights: Event weights
    @param other_background: Other backgrounds constants
    @param verbose: Turn on displaying information about computation
    @return: List containing efficiencies, S, B, and two significances (norml & simplified)
    """

    cm = confusion_matrix(true_y, predicted_y, sample_weight=weights)
    cm_len = np.shape(cm)[0]
    signal_total = cm[0, 0] #0,0
    background_total = 0
    for i in range(1, cm_len): # no -1
        background_total += cm[i, 0]

    S = signal_total * test_set_scale_factor
    B = background_total * test_set_scale_factor
    B += other_background
    if B < 1:
        S = 0
        B = 1

    signif = S / np.sqrt(S + B)
    signif_simple = S / np.sqrt(B)
    signif_improved = S / (np.sqrt(B) + 3/2)
    if signif_simple == float('+inf'):
        signif_simple = 0
    result = {'S': S,
              'B': B,
              'significance': signif,
              'significance_simple': signif_simple,
              'significance_improved': signif_improved}
    return result

def plot_threshold(x_values, y_values, optimums, title, ylabel, colors, labels, savepath: str,
                   force_threshold_value=None):
    """ Plots graphs for threshold characteristics.
    @param x_values: Values for x-axis (list)
    @param y_values: Values for y-axis (list of lists)
    @param optimums: Whether to search max or min values in y_values
    @param title: Title of plot
    @param colors: Colors for y_values
    @param labels: Labels for y_values
    @param force_threshold_value: Value of forced vertical line value (if None, vertical line is computed with usage of
    optimums parameter)
    """
    best_score_final = 0
    plt.figure()
    for values, optimum, color, label in zip(y_values, optimums, colors, labels):
        plt.plot(x_values, values, color=color, label=label, linewidth=2)

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

        # Plot a dotted vertical line at the best score for that scorer marked by x
        plt.plot([x_values[best_index], ] * 2 , [0, best_score],
                linestyle='-.', color=color, marker='x', markeredgewidth=3, ms=8)

        # Annotate the best score for that scorer
        plt.annotate("%0.3f" % best_score,
                    (x_values[best_index], best_score + 0.005))
        if best_score > best_score_final:
            best_score_final = best_score

    plt.xticks(np.arange(0.0, 1.0, step=0.1))
    plt.xlabel('Threshold')
    plt.ylabel(ylabel)
    # if len(y_values) == 2:
    #     plt.yscale('log')
    plt.title(title)
    plt.legend(loc="best")
    plt.savefig(savepath)
    # plt.show()
    # plt.close()
    return best_score_final


def main():
    classify()

if __name__ == "__main__":
    main()