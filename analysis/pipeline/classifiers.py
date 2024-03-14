"""
@author: Jakub Maly, Lukas Vicenik
Support methods for ML libraries
"""
from cProfile import label
import sys
from turtle import left
sys.path.append("/media/lucas/ADATA SE760/KYR/bc_thesis/thesis/project/")
sys.path.append("/media/lucas/ADATA SE760/KYR/bc_thesis/thesis/project/load/")
sys.path.append("/media/lucas/ADATA SE760/KYR/bc_thesis/thesis/project/utils/")
sys.path.append("/media/lucas/ADATA SE760/KYR/bc_thesis/thesis/project/plot/analysis/")
from typing import List, Dict, Tuple
from easydict import EasyDict
import config.constants as C
import root_load.root_reader as reader
import seaborn as sns
import pickle
import compress_pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
from sklearn.pipeline import Pipeline
import csv
import random
import logging

from feature_imp_sum import feature_imp_sum

from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from xgboost.sklearn import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import auc

import confmatrix_prettyprint as com
import logging
from sklearn.model_selection import GridSearchCV


def load_numpyarray(filepath: str) -> np.array:
    with open(filepath, 'rb') as f:
     return np.load(f)
def save_numpyarray(file: np.array, filepath: str):
    with open(filepath, 'rb') as f:
     np.save(f, file)
def read_X_y_as_DataFrame(X, y):
    pass
def save(f, folder, name):
    """ Saves file.
    @param f: File to be saved
    @param folder: Folder for saving
    @param name: Name of save file
    """
    pickle.dump(f, open(folder + '/' + name + '.pkl', 'wb'), protocol=4)
def save_compress(f, folder, name, method):
    """ Compress and saves file.
    @param f: File to be saved
    @param folder: Folder for saving
    @param name: Name of save file
    @param method: Compression method
    """
    compress_pickle.dump(f, open(folder + '/' + name + '.pkl.' + method, 'wb'), compression=method)
def save_as_npy(array: np.array, folder: str, name: str):
    with open(folder + '/' + name + ".npy", 'wb') as f:
        np.save(f,  array)
def load(f):
    """ Loads file.
    @param f: File to be loaded
    @return: Content of file
    """
    return pickle.load(open(f, 'rb'))
def load_compress(f, method):
    """ Loads compressed file.
    @param f: File to be loaded
    @param method: Compression method
    @return: Content of file
    """
    return compress_pickle.load(open(f, 'rb'), method)

def multiclass_roc_auc_score(y_test, y_pred, average="macro", multi_class='ovr'):
    """ Computes ROC/AUC score for multiclass classification
    @param y_test: test labels
    @param y_pred: predicted labels
    @param average: averaging algorithm
    @param multi_class: how to handle multiclass
    @return: ROC/AUC score
    """
    # lb = LabelBinarizer()
    # lb.fit(y_test)
    # y_test = lb.transform(y_test)
    # y_pred = lb.transform(y_pred)
    # print(y_test)
    # print(y_pred)
    return roc_auc_score(y_test, y_pred, average=average, multi_class=multi_class)
def multiclass_f1(y_test, y_pred, average="macro", ):
    """ Computes F-1 score for multiclass classification
    @param y_test: test labels
    @param y_pred: predicted labels
    @param average: averaging algorithm
    @return: F-1 score
    """
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)
    return f1_score(y_test, y_pred, average=average)
def multiclass_accuracy(y_test, y_pred):
    """ Computes accuracy score for multiclass classification
    @param y_test: test labels
    @param y_pred: predicted labels
    @return: Accuracy score
    """
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)
    return accuracy_score(y_test, y_pred)

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


def train_model(X, y, w, name=None, parameters={}, svd_components=0):
    """ Trains classifier with the data. Also performs basic scaling of data.
    @param X: Data
    @param y: Ground truth labels
    @param w: weights by sample
    @param name: Name of the classifier (None, RFC, KNC, GNB, ADA, GBC)
    @param parameters: Parameters for given model
    @param svd_components: Number of components for Truncated SVD
    @return: Scikit-learn pipeline
    """

    print('ml_utils.train_model():')

    # Scaler
    sc = StandardScaler()

    if name == 'XGB':
        grid = {'classifier': 'xgb', 'max_depth': 4, 'gamma': 0,
        'learning_rate': 0.01, 'n_estimator': 80,
        'min_child_weight': 0.01, 'subsample': 0.5, 'colsample_bytree': 0.9,
        'scale_pos_weight': 7, 'seed': 23,'num_class': 7,
            'tree_method': 'gpu_hist'}

        # grid_empty = {
        #     'num_class': 7,#
        #     'tree_method': 'gpu_hist',#
        #     'random_state': 42,#
        #
        # }
        grid_empty = {
            'colsample_bytree': 1.0, 'gamma': 0.00025, 'learning_rate': 0.5, 'max_depth': 6, 'min_child_weight': 0.043,
            'n_estimators': 2300, 'reg_alpha': 0.0036000000000000003, 'scale_pos_weight': 10, 'subsample': 0.8,
            'num_class': 7,
            'tree_method': 'gpu_hist',
            'random_state': random.randint(0, 1000)

        }
        clf = XGBClassifier()

        clf.set_params(**grid_empty)

        # XGBoost clssifier
        print('\t|-> training XGBoostClassifier')
        pipe = Pipeline([
            ('standard_scaler', sc), 
        #  ('pca', PCA()), 
            ('clf', clf)
        ])

        #pipe.fit(X, y, **{'clf__sample_weight': w})
        pipe.fit(X, y)


    return pipe


def predict(model, X):
    """ Predicts outputs for model (pipeline).
    @param model: Scikit-learn model (pipeline)
    @param X: Inputs for prediction
    @return: Predicted classes
    """
    
    return model.predict(X)


def predict_proba(model, X):
    """ Predicts outputs for model (pipeline).
    @param model: Scikit-learn model (pipeline)
    @param X: Inputs for prediction
    @return: Predicted class probabilities
    """

    return model.predict_proba(X)


def plot_histograms_multiclass(X, y, classes, feature, bins=None, saving=False, save_folder=None):
    """ Plots histograms with kernel density estimation for multiclass data.
    @param X: Data
    @param y: Ground truth labels
    @param classes: Dictionary with classes
    @param feature: Name of feature for which histogram should be plotted
    @param bins: Number of bins (None will force automatic selection)
    @param saving: Whether to save plot or no
    @param save_folder: Where to save plot (only when saving is True)
    """

    #Determine if values are integers
    integers = True
    values = set(X)
    for value in values:
        if value - int(value) != 0:
            integers = False
    if integers == True:
        #print("Calculated number of bins: " + str(max(values) - min(values)))
        bins = int(max(values) - min(values)) + 1


    plt.figure()
    plt.title(feature)
    for key in classes:
        # sns.distplot(X[y == key], bins, hist_kws={'histtype': 'step'}, kde_kws={'label': '%s (KDE)' % classes[key]},
        # label='%s' % classes[key])
        print("this is X[y == key]: " + str(X[y == key]))
      
                
        sns.distplot(X[y == key], bins, hist_kws={'histtype': 'step', "linewidth": 1}, kde=False,
                     label='%s' % classes[key])

    plt.legend(loc="best")

    if saving:
        plt.savefig(save_folder + feature + '.png')
        plt.close()


def plot_pred_hist_multiclass(y_preds, classifiers, classes, colors):
    """ Plots predict histograms for multi-class classification.
    @param y_preds: List of predictions for classifiers
    @param classifiers: List of classifiers
    @param classes: Dictionary of classes
    @param colors: Colors used for classes
    """

    for i, y_pred in enumerate(y_preds):
        y_pred = np.array(y_pred)
        heights = list()
        labels = list()

        # Compute heights and note labels
        for key in classes:
            heights.append(sum(y_pred == key))
            labels.append(classes[key])

        plt.figure()
        x = np.arange(3)
        barlist = plt.bar(x, height=heights)

        # Set colors
        for j, bar in enumerate(barlist):
            bar.set_color(colors[j])

        plt.xticks(x, labels)
        plt.title('{} predictions'.format(classifiers[i]))




def plot_feat_imp(title, model, f, limit, folder, classifier, verbose=True):
    """ Plots feature importances for model containing RFC classificator.
    @param title: Title of plot
    @param model: RFC classificator or pipeline containing one
    @param f: List of features
    @param limit: Number of features to be plot
    @param verbose: Whether lor importances to console
    """
    if classifier == "TABNET":
        imp = np.round(model.feature_importances_, 5)
    else:
        imp = np.round(model['clf'].feature_importances_, 5)
    #std = np.round(np.std([tree.feature_importances_ for tree in model['clf'].estimators_], axis=0), 5)
    ind = np.argsort(imp)[::-1]

    imp_plot = list()
    ind_plot = list()
    features_limit = list()

    if verbose:
        print("\nFeatures importance:")
    dataset_folder_2 = "data_processed"
    EXPERIMENTS = "/media/lucas/ADATA SE760/KYR/bc_thesis/" + dataset_folder_2 + "/final_data_analysis_weights/experiments/"
    logger = logging.getLogger('signifs_logger')
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(EXPERIMENTS + '/feature_sum.log')
    fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)
    # create formatter and add it to the handlers
    #formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    # ch.setFormatter(formatter)
    # fh.setFormatter(formatter)
    # add the handlers to logger
    logger.addHandler(ch)
    logger.addHandler(fh)
    for i in range(limit):
        imp_plot.append(imp[ind[i]])
      #  std_plot.append(std[ind[i]])
        ind_plot.append(ind[i])
        features_limit.append(f[ind[i]])
        # if verbose:
        #     print(f[ind_plot[-1]] + ', imp: {}'.format(imp_plot[-1]))
        #     feature_imp_sum(f[ind_plot[-1]], i, logger)
            #print(f[ind_plot[-1]] + ', imp: {}, std: {}'.format(imp_plot[-1], std_plot[-1]))

    imp_plot.reverse()
    plt.figure()
    plt.title(title)
    plt.barh(range(limit),imp_plot, label="Feature importance")
    plt.ylim([-1, limit])
    plt.subplots_adjust(left=0.4)
    plt.yticks(range(limit), features_limit, rotation = 0)
    plt.title('Store Inventory')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.savefig(folder+classifier+"_feature_importance.png")
    with open(folder+classifier+"_features.csv", "w+") as fil:
        writer = csv.writer(fil)
        writer.writerows(map(lambda x: [x], features_limit))

def check_feature_order(X, y, w, f, f_prev):
    """ Checks if features are in desired order
    @param X: Data
    @param y: Ground truth Labels
    @param f: Current order
    @param f_prev: Previous (desired) order
    @return: Reordered data and labels, desired feature order
    """

    X_new = X
    y_new = y
    w_new = w

    for feature in f_prev:
        i = f_prev.index(feature)
        j = f.index(feature)

        if i != j:
            print('features columns were exchanged')
            X_new[:, i] = X[:, j]
            y_new[i] = y[j]
            w_new[i] = w[j]

    return X_new, y_new, w_new, f_prev

def threshold_probas_multiclass(y_probas, threshold):
        """ Thresholds multiclass probabilities - see thesis (WP tuning) for more information
        @param y_probas: Predicted probabilities
        @param threshold: Probability threshold
        @return: Predicted classes
        """

        length = np.shape(y_probas)[0]
        width = np.shape(y_probas)[1]

        y_pred = list()

        for i in range(length):
            bgr_sum = 0
            for j in range(1, width):
                bgr_sum += y_probas[i, j]

            if y_probas[i, 0] >= threshold:
                y_pred.append(0)
            else:
                suggestion = 0
                max_p = 0
                if suggestion == 0:
                    for j in range(1, width):
                        if max_p < y_probas[i, j]:
                            max_p = y_probas[i, j]
                            suggestion = j
                y_pred.append(suggestion)

        return y_pred


def threshold_characteristics_multiclass_significance(y_probas, y_test, weights, total, other_bgr):
    """ Obtain characteristics for simplified significance maximization threshold tuning
    @param y_probas: Predicted probabilities
    @param y_test: Ground truth labels
    @param weights: Weighting factors used for comparison with existing results
    @param total: Number of events in real data used for comparison
    @param other_bgr: Number of events in other background in real data used for comparison
    @return: Graph data, best threshold
    """

    x_values = list()

    y_eps_0 = list()
    y_eps_1 = list()
    y_eps_2 = list()
    y_S = list()
    y_B = list()
    y_signif = list()
    y_signif_simp = list()

    max_sig = 0
    best_tr = 0

    for tr in np.round(np.arange(0.0, 1, 0.01), 2):
        tr = np.round(tr, 3)
        x_values.append(tr)
        print('Threshold: {}'.format(tr))
        y_pred = threshold_probas_multiclass(y_probas, tr)
        response = significance_score_multiclass_compare(y_test, y_pred, weights, total, other_bgr, False)
        y_eps_0.append(response['eps'][0])
        y_eps_1.append(response['eps'][1])
        y_eps_2.append(response['eps'][2])
        y_S.append(response['S'])
        y_B.append(response['B'])
        y_signif.append(response['signif'])
        y_signif_simp.append(response['signif_simp'])

        if response['signif_simp'] > max_sig:
            max_sig = response['signif_simp']
            best_tr = tr

    return x_values, y_eps_0, y_eps_1, y_eps_2, y_S, y_B, y_signif, y_signif_simp, best_tr



def threshold_characteristics_multiclass_sensitivity(y_probas, y_test):
    """ Obtain characteristics for sensitivity maximization threshold tuning
    @param y_probas: Predicted probabilities
    @param y_test: Ground truth labels
    @return: Graph data, best threshold
    """

    x_values = list()

    y_fdr = list()
    y_sen = list()

    max_sig = 0
    best_tr = 0

    for tr in np.arange(0, 1.01, 0.01):
        tr = np.round(tr, 3)
        x_values.append(tr)
        print('Threshold: {}'.format(tr))
        y_pred = threshold_probas_multiclass(y_probas, tr)
        fdr = 1 - precision_score(y_test, y_pred, average='weighted')
        sen = recall_score(y_test, y_pred, average='weighted')

        y_fdr.append(fdr)
        y_sen.append(sen)

        if sen > max_sig:
            max_sig = sen
            best_tr = tr

    return x_values, y_fdr, y_sen, best_tr


def significance_score_multiclass_compare(y, y_pred, weights={}, total={}, compensation=0, verbose=True):
    """ Approximate significance score for prediction.
        Used to compare with known results for given weights, number of events, and background compensation constant.
    @param y: Ground truth labels
    @param y_pred: Predicted probabilities
    @param weights: Class weights
    @param total: Number of events in given class
    @param compensation: Other backgrounds constants
    @param verbose: Turn on displaying information about computation
    @return: List containing efficiencies, S, B, and two significances (norml & simplified)
    """

    # Selected
    cm = confusion_matrix(y, y_pred)
    cm_len = np.shape(cm)[0]

    # Efficiencies
    sig_total = cm[0, 0]
    epsilon_bgrs = list()
    bgr_sums = list()
    for i in range(1, cm_len):
        sig_total += cm[0, i]
        epsilon_bgrs.append(cm[i, 0])
        bgr_sums.append(sum(cm[i, j] for j in range(cm_len)))
    epsilon_sig = cm[0, 0] / sig_total
    for i in range(len(epsilon_bgrs)):
        epsilon_bgrs[i] = epsilon_bgrs[i] / bgr_sums[i]

    S = epsilon_sig * total[0] * weights[0]
    B = 0
    for i in range(len(epsilon_bgrs)):
        B += epsilon_bgrs[i] * total[i + 1] * weights[i + 1]
    B += compensation

    if verbose:
        print('ml_utils.sig_score():')
        print('\t|-> eps 0: %0.4f' % epsilon_sig)
        for i in range(len(epsilon_bgrs)):
            print('\t|-> eps %d: %0.4f' % (i + 1, epsilon_bgrs[i]))
        print('\t|-> S: %0.4f' % S)
        print('\t|-> B: %0.4f' % B)

    signif = S / np.sqrt(S + B)
    signif_simp = S / np.sqrt(B)
    eps = list()
    eps.append(epsilon_sig)
    for i in range(len(epsilon_bgrs)):
        eps.append(epsilon_bgrs[i])

    ret_dic = {'eps': eps,
               'S': S,
               'B': B,
               'signif': signif,
               'signif_simp': signif_simp}

    return ret_dic


def print_class_frequencies_in_dataset(y : np.array, name : str, logger : logging.Logger):
    logger.info("Dataset {}".format(name))
    for i in range(7):
        logger.info("\tClass {} - {} samples = {} %".format(i, len(np.where(y==i)[0]),len(np.where(y==i)[0])*100/len(y)))



def visualize_network_training_progress( train_losses : list, valid_losses : list, path: str,
                                        save : bool = True, show : bool = False):
    x_labels = np.arange(1,len(train_losses)+1)
    plt.plot(x_labels,train_losses, label='Training loss')
    plt.plot(x_labels,valid_losses, label='Validation loss')
    plt.xticks(np.arange(1,len(train_losses)+1,2))
    plt.xlabel("Number of Epochs")
    plt.ylabel("Loss")
    plt.title("Development of the losses during training", loc='left')
    plt.legend()
    if save and path is not None:
        plt.savefig(path + "imgs/network_training_progress_mlp.png")
    if show:
        plt.show()
    plt.close()

def visualize_network_accuracy_progress(train_acc: list, valid_acc: list,path: str,
                                        save : bool = True, show : bool = False):
    x_labels = np.arange(1, len(train_acc) + 1)
    plt.plot(x_labels,train_acc, label='Training Accuracy')
    plt.plot(x_labels,valid_acc, label='Validation Accuracy')
    plt.xticks(np.arange(1,len(train_acc)+1,2))
    plt.xlabel("Number of Epochs")
    plt.ylabel("Accuracy")
    plt.title("Development of the accuracy during training", loc='left')
    plt.legend()
    if save and path is not None:
        plt.savefig(path + "imgs/network_accuracy_progress_mlp.png")
    if show:
        plt.show()
    plt.close()

# Helper method to print importances and visualize distribution
def visualize_importances(feature_names, importances, title="Average Feature Importances", plot=True, axis_title="Features", config: EasyDict = None):
    for i in range(len(feature_names)):
        print(feature_names[i], ": ", '%.3f'%(importances[i]))
    tuples = [(feature_names[i], importances[i]) for i in range(len(importances))]
    tuples = sorted(tuples, key=lambda x: x[1])
    tuples = tuples[-10:] + tuples[:10]
    tuples = sorted(tuples, key=lambda x: x[1]*(-1))
    x_pos = (np.arange(len(tuples)))
    if plot:
        plt.figure(figsize=(12, 6))
        plt.bar(x_pos, [t[1] for t in tuples], align='center')
        plt.xticks(x_pos, [t[0] for t in tuples], rotation='vertical')
        plt.xlabel(axis_title)
        plt.title(title)
    if save:
        plt.savefig(config.figures_dir + "feature_importances.png", bbox_inches="tight")
    plt.close()


def make_cmap(colors, position=None, bit=False):
    '''
    This wonderful method was taken from http://schubert.atmos.colostate.edu/~cslocum/custom_cmap.html
    @author Chris Slocum

    make_cmap takes a list of tuples which contain RGB values. The RGB
    values may either be in 8-bit [0 to 255] (in which bit must be set to
    True when called) or arithmetic [0 to 1] (default). make_cmap returns
    a cmap with equally spaced colors.
    Arrange your tuples so that the first color is the lowest value for the
    colorbar and the last is the highest.
    position contains values from 0 to 1 to dictate the location of each color.
    '''

    bit_rgb = np.linspace(0, 1, 256)
    if position == None:
        position = np.linspace(0, 1, len(colors))
    else:
        if len(position) != len(colors):
            sys.exit("position length must be the same as colors")
        elif position[0] != 0 or position[-1] != 1:
            sys.exit("position must start with 0 and end with 1")
    if bit:
        for i in range(len(colors)):
            colors[i] = (bit_rgb[colors[i][0]],
                         bit_rgb[colors[i][1]],
                         bit_rgb[colors[i][2]])
    cdict = {'red': [], 'green': [], 'blue': []}
    for pos, color in zip(position, colors):
        cdict['red'].append((pos, color[0], color[0]))
        cdict['green'].append((pos, color[1], color[1]))
        cdict['blue'].append((pos, color[2], color[2]))

    cmap = mpl.colors.LinearSegmentedColormap('my_colormap', cdict, 256)
    return cmap
