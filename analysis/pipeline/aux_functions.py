import os
import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt

from sklearn.preprocessing import label_binarize
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, auc, roc_curve
import umap
from sklearn.manifold import TSNE
import csv

# def plot_output(y_pred_proba, y_test):
#     print(y_pred_proba)
#     print(y_test)
#     for prob, label in zip(y_pred_proba, y_test):
#         if label == 1:
            
#         elif label == 0:

def mass_to_num(mass):
    num = {
        'tbh_250_new': 1,
        'tbh_800_new': 2,
        'tbh_3000_new': 3,
        'tbh_300': 4,
        'tbh_800': 5,
        'tbh_1500': 6,
        'tbh_2000': 7
        }[mass]
    return num

def num_to_mass(num):
    mass = {
        1: 'tbh_250_new',
        2: 'tbh_800_new',
        3: 'tbh_3000_new',
        4: 'tbh_300',
        5: 'tbh_800',
        6: 'tbh_1500',
        7: 'tbh_2000'
    }[num]
    return mass
    

def plot_ouput(weight, y_true, y_probs, PATH_RESULTS, classifier_type):
    
    if not os.path.exists(f'{PATH_RESULTS}'):
        os.makedirs(f'{PATH_RESULTS}')
    
    df_weight_prob = pd.DataFrame()
    df_weight_prob.insert(0, 'prob', y_probs, True)
    df_weight_prob.insert(1, 'weight', weight.values, True)
    df_weight_prob.insert(2, 'y', y_true.values, True)
    bins_sig = []
    bins_bkg = []
    
    df_weight_prob_sig =  df_weight_prob[df_weight_prob['y'] == 0]
    df_weight_prob_bkg =  df_weight_prob[df_weight_prob['y'] == 1]

    ths = []
    for idx in range(0, 20 ,1):
        th = idx*0.05
        bin_sig = df_weight_prob_sig[(df_weight_prob_sig['prob'] < th + 0.05) & (df_weight_prob_sig['prob'] > (th))]['weight'].sum()
        bin_bkg = df_weight_prob_bkg[(df_weight_prob_bkg['prob'] < th + 0.05) & (df_weight_prob_bkg['prob'] > (th))]['weight'].sum()
        bins_sig.append(bin_sig)
        bins_bkg.append(bin_bkg)
        ths.append(th)
    
    y_probs_sig = y_probs[df_weight_prob['y'] == 0]
    y_probs_bkg = y_probs[df_weight_prob['y'] == 1]

    # plt.bar(ths, bins_sig, color='dodgerblue',edgecolor='black', width=0.05, align='edge', alpha=0.7, label='Signal')
    # plt.bar(ths, bins_bkg, color='mediumvioletred',edgecolor='black', width=0.05, align='edge', alpha=0.7, label='Background')
    
    plt.clf()
    plt.figure(figsize=(10, 8))
    plt.hist(y_probs_sig, bins=20, color='royalblue', edgecolor='black', alpha=0.7, label='Signal')  
    plt.hist(y_probs_bkg, bins=20, color='crimson', edgecolor='black', alpha=0.7, label='Background') 
    plt.grid()
    plt.legend(fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel('Classifier threshold', fontsize=20)
    plt.ylabel('Number of events', fontsize=20)
    #plt.title('Outputs of the classifier for Signal and Background', fontsize=18)
    plt.grid(True)
    
    plt.savefig(f'{PATH_RESULTS}{classifier_type}/ouput_hist_normal.pdf')
    
    plt.clf()
    plt.figure(figsize=(10, 8))
    plt.bar(ths, bins_sig, color='dodgerblue',edgecolor='black', width=0.05, align='edge', alpha=0.7, label='Signal')
    plt.bar(ths, bins_bkg, color='mediumvioletred',edgecolor='black', width=0.05, align='edge', alpha=0.7, label='Background')
    plt.grid()
    plt.legend(fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel('Classifier threshold', fontsize=20)
    plt.ylabel('Number of events', fontsize=20)
    #plt.title('Outputs of the classifier for Signal and Background', fontsize=18)
    plt.grid(True)

    plt.savefig(f'{PATH_RESULTS}{classifier_type}/ouput_hist_weighted.pdf')

def plot_roc_multiclass(title, y, y_probas, classes, scores, folder):
    """ Plots ROC curve for multi-class classification.
    @param title: Title of plot
    @param y: True labels
    @param y_probas: Predicted probabilities
    @param classes: Dictionary with classes
    @param scores: Dictionary with scores
    """
    plt.clf()
    y_probas[:, [1, 0]] = y_probas[:, [0, 1]]
    y_bin = label_binarize(y, classes=list(classes.keys()))
    n_classes = y_bin.shape[1]
    fprs = dict()
    tprs = dict()
    aucs = dict()
    for i in range(n_classes):
        fprs[i], tprs[i], _ = roc_curve(y_bin[:, i], y_probas[:, i])
        aucs[i] = auc(fprs[i], tprs[i])
    #plt.figure()
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
    y_probas[:, [1, 0]] = y_probas[:, [0, 1]]
    plt.grid()
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
    plt.clf()
    x_values = list()
    y_S = list()
    y_B = list()
    y_signif = list()
    y_signif_simp = list()
    y_signif_imp = list()
    y_signif_true = list()
    y_BB = list()
    max_sig = 0
    best_th = 0
    threshold_start = 0
    logger = logging.getLogger()
    for th in np.round(np.arange(threshold_start, 1, 0.01), 2):
        th = np.round(th, 3)
        x_values.append(th)
        y_pred = calculate_class_predictions_basedon_decision_threshold(predicted_probs_y, th)
        response = calculate_significance_one_threshold_new_method(true_y, y_pred, weights, other_bgr,
                                                                   test_set_scale_factor)
        
        y_S.append(response['S'])
        y_B.append(response['B'])
        y_signif_true.append(response['significance_true'])
        y_signif_simp.append(response['significance_simple'])
        y_signif.append(response['significance'])
        y_signif_imp.append(response['significance_improved'])
        
        y_BB.append(response['BB'])

        if response['significance_true'] > max_sig:
            max_sig = response['significance_true']
            best_th = th
    return x_values, y_S, y_B, y_signif_true,y_signif_simp, y_signif,y_signif_imp,  best_th, y_BB

def calculate_class_predictions_basedon_decision_threshold(predicted_probs_y: np.array, threshold: float):
    """ Outputs an array with predicted classes based on predicted class probabilities and decision threshold
    If the predicted probability for class 0 is greater than threshold, the predicted class is 0,
    otherwise it is max P(y|x) among the remaining classes
    @param predicted_probs_y: Predicted probabilities
    @param threshold: Decision threshold
    @return: Predicted classes

    """
    plt.clf()
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
    plt.clf()
    cm = confusion_matrix(true_y, predicted_y, sample_weight=weights)
    cm_len = np.shape(cm)[0]
    signal_total = cm[0, 0] #0,0
    background_total = 0
    for i in range(1, cm_len): # no -1
        background_total += cm[i, 0]
    S = signal_total * test_set_scale_factor
    B = background_total * test_set_scale_factor
    B += other_background
    if B < 0.1:
        S = 0
        B = 0.1

    BB = cm[1,1]

    signif = S / np.sqrt(S + B)
    signif_simple = S / np.sqrt(B)
    signif_improved = S / (np.sqrt(B) + 3/2)
    signif_true = np.sqrt(2*(S+B)*np.log2(1+S/B)-2*S)
    
    if signif_simple == float('+inf'):
        signif_simple = 0
    result = {'S': S,
              'B': B,
              'significance_true': signif_true,
              'significance_simple': signif_simple,
              'significance': signif,
              'significance_improved': signif_improved,
              'BB': BB}
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
    plt.clf()
    plt.figure(figsize=(10, 8))
    best_score_final = 0
    best_scores_final = []
    #plt.figure()
    for idx, (values, optimum, color, label) in enumerate(zip(y_values, optimums, colors, labels)):
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
        
        plt.plot([x_values[best_index], ] * 2 , [0, best_score],
                linestyle='-.', color=color, marker='x', markeredgewidth=3, ms=8)

        plt.annotate("%0.3f" % best_score,
                    (x_values[best_index], best_score + 0.005),fontsize=20)
        if len(y_values) == 4:    
            if idx == 0 or idx == 1:
                best_scores_final.append(best_score)
        else:
            if best_score > best_score_final:
                best_scores_final.append(best_score)
    
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xticks(np.arange(0.0, 1.0, step=0.1))
    plt.xlabel('Threshold', fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    # if len(y_values) == 2:
    #     plt.yscale('log')
    #plt.title(title)
    plt.legend(loc="best",fontsize=20)
    plt.grid()
    plt.savefig(savepath)
    # plt.show()
    # plt.close()
    #print(best_scores_final)
    if len(y_values) == 4:
        return best_scores_final
    else:
        return best_scores_final

def plot_feature_importnace(PATH_DATA, model, X_train):
    plt.clf()
    FEATURE_NUM = 30
    importances = model.named_steps['clf'].feature_importances_

    indices = np.argsort(importances)[::-1][:FEATURE_NUM]  

    #plt.figure(figsize=(10, 6))
    plt.title("Top 20 Feature Importances")
    plt.bar(range(FEATURE_NUM), importances[indices], align="center")
    plt.xticks(range(FEATURE_NUM), X_train.columns[indices], rotation=90)
    plt.xlim([-1, FEATURE_NUM])
    plt.tight_layout()
    plt.savefig(f'{PATH_DATA}feature_importance_top20.png')
    top_features = X_train.columns[indices]
    PATH = '/home/lucas/Documents/KYR/msc_thesis/vae-generator-for-particle-physics/analysis/logging/'
    FILE = f'features_top_{len(top_features)}.log'
    top_features = list(top_features)
    with open(PATH + FILE, 'a') as file:
        file.write('[')
        for idx, top_feature in enumerate(top_features):
            if idx != len(top_features)-1:
                file.write(f'"{top_feature}", ')
            else: file.write(f'"{top_feature}"')
        file.write(']')
        file.write('\n')
    

def compute_embed(simulated_list, generated_list, TYPE, sample_size=3000):
    plt.clf()
    PATH_DATA = '/home/lucas/Documents/KYR/msc_thesis/vae-generator-for-particle-physics/analysis/results/metrics/'
    print("RUNNING t-SNE")

    random_state = 42
    
    simulated_list = simulated_list[np.random.choice(simulated_list.shape[0], sample_size, replace=False)]
    generated_list = generated_list[np.random.choice(generated_list.shape[0], sample_size, replace=False)]
    
    t_sne_instance = TSNE(n_components=2, random_state=random_state)
    
    n_simulated = len(simulated_list)
    n_generated = len(generated_list)
    
    lists = np.concatenate((simulated_list, generated_list))
    trans_lists = t_sne_instance.fit_transform(lists)
    
    trans_simulated = trans_lists[:n_simulated]
    trans_generated = trans_lists[n_simulated:]
    
    np.savetxt(f'{PATH_DATA}points_simulated_{TYPE}.csv', trans_simulated, delimiter=',')
    np.savetxt(f'{PATH_DATA}points_generated_{TYPE}.csv', trans_generated, delimiter=',')
        
def visualize_embed(TYPE):
    options = {'std': 'Standard ELBO', 'sym': 'Standard SEL', 'std_h' : 'Ladder ELBO', 'sym_h' : 'Ladder SEL'}
    PATH_DATA = '/home/lucas/Documents/KYR/msc_thesis/vae-generator-for-particle-physics/analysis/results/metrics/'

    files = [f'points_simulated_{TYPE}.csv', f'points_generated_{TYPE}.csv']
    
    points_simulated = np.genfromtxt(f'{PATH_DATA}{files[0]}', delimiter=',')
    points_generated = np.genfromtxt(f'{PATH_DATA}{files[1]}', delimiter=',')

    embed_simulated_df = pd.DataFrame(data=points_simulated, columns=["t-SNE Component 1", "t-SNE Component 2"])
    embed_simulated_df["Source"] = "Simulated Data"
    embed_generated_df = pd.DataFrame(data=points_generated, columns=["t-SNE Component 1", "t-SNE Component 2"])
    embed_generated_df["Source"] = f"Generated {TYPE}"

    plt.clf()
    plt.figure(figsize=(10, 8))
    plt.scatter(embed_simulated_df["t-SNE Component 1"], embed_simulated_df["t-SNE Component 2"], c='blue', label='Simulated Data', s=3, alpha=1.0)
    plt.scatter(embed_generated_df["t-SNE Component 1"], embed_generated_df["t-SNE Component 2"], c='red', label='Generated Data', s=3, alpha=1.0)
    plt.title(r"Data space $\mathcal{X}$" + f' - {options[TYPE]}', fontsize=36)
    plt.xlabel("t-SNE Component 1", fontsize=28) 
    plt.ylabel("t-SNE Component 2", fontsize=28)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    legend2 = plt.legend(fontsize=28, scatterpoints=20)  
    legend2.get_frame().set_alpha(0.8)
    plt.grid(True)
    
    plt.savefig(f'{PATH_DATA}vae_data_tsne_results_{TYPE}.pdf')
    print("t-SNE VISUALIZATION SAVED")