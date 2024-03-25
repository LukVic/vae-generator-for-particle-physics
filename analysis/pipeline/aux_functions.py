
import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt

from sklearn.preprocessing import label_binarize
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, auc, roc_curve
import umap


# def plot_output(y_pred_proba, y_test):
#     print(y_pred_proba)
#     print(y_test)
#     for prob, label in zip(y_pred_proba, y_test):
#         if label == 1:
            
#         elif label == 0:

def plot_roc_multiclass(title, y, y_probas, classes, scores, folder):
    """ Plots ROC curve for multi-class classification.
    @param title: Title of plot
    @param y: True labels
    @param y_probas: Predicted probabilities
    @param classes: Dictionary with classes
    @param scores: Dictionary with scores
    """
    y_probas[:, [1, 0]] = y_probas[:, [0, 1]]
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
    y_probas[:, [1, 0]] = y_probas[:, [0, 1]]
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
    y_BB = list()
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
        y_BB.append(response['BB'])

        if response['significance_simple'] > max_sig:
            max_sig = response['significance_simple']
            best_th = th
    return x_values, y_S, y_B, y_signif, y_signif_simp, y_signif_imp, best_th, y_BB

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

    BB = cm[1,1]

    signif = S / np.sqrt(S + B)
    signif_simple = S / np.sqrt(B)
    signif_improved = S / (np.sqrt(B) + 3/2)
    if signif_simple == float('+inf'):
        signif_simple = 0
    result = {'S': S,
              'B': B,
              'significance': signif,
              'significance_simple': signif_simple,
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

def plot_feature_importnace(PATH_DATA, model, X_train):
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
    

def compute_embed(restricted_list, loosened_list, generated_list):
    PATH_DATA = '/home/lucas/Documents/KYR/msc_thesis/vae-generator-for-particle-physics/analysis/data/common/'
    print("RUNNING UMAP")
    random_state = 42
    #embed_instance = embed(n_components=2, perplexity=30, n_jobs=12, random_state=random_state)
    umap_instance = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1)
    
    n_restricted = len(restricted_list)
    n_loosened = len(loosened_list)
    n_generated = len(generated_list)
    
    lists = list(restricted_list) + list(loosened_list) + list(generated_list)
    trans_lists = umap_instance.fit_transform(np.array(lists))
    
    trans_restricted = trans_lists[:n_restricted]
    trans_loosened = trans_lists[n_restricted:(n_restricted + n_loosened)]
    trans_generated = trans_lists[(n_restricted + n_loosened):]
    
    #trans_data = trans_feature
    
    np.savetxt(f'{PATH_DATA}points_restricted.csv',trans_restricted, delimiter=',')
    np.savetxt(f'{PATH_DATA}points_loosened.csv',trans_loosened, delimiter=',')
    np.savetxt(f'{PATH_DATA}points_generated.csv',trans_generated, delimiter=',')
        
def visualize_embed(PATH_DATA, PATH_RESULTS):
    files = ['points_restricted.csv', 'points_loosened.csv', 'points_generated.csv']
    
    points_restricted = np.genfromtxt(f'{PATH_DATA}{files[0]}', delimiter=',')
    points_loosened = np.genfromtxt(f'{PATH_DATA}{files[1]}', delimiter=',')
    points_generated = np.genfromtxt(f'{PATH_DATA}{files[2]}', delimiter=',')

    # sample_size = 3000
    # points_restricted = points_restricted[np.random.choice(points_restricted.shape[0], sample_size, replace=False)]
    # points_loosened = points_loosened[np.random.choice(points_loosened.shape[0], sample_size, replace=False)]
    # points_generated = points_generated[np.random.choice(points_generated.shape[0], sample_size, replace=False)]
    
    embed_restricted_df = pd.DataFrame(data=points_restricted, columns=["UMAP_1", "UMAP_2"])
    embed_restricted_df["Source"] = "Decoded Data ELBO"
    embed_loosened_df = pd.DataFrame(data=points_loosened, columns=["UMAP_1", "UMAP_2"])
    embed_loosened_df["Source"] = "Decoded Data SYMM"
    embed_generated_df = pd.DataFrame(data=points_generated, columns=["UMAP_1", "UMAP_2"])
    embed_generated_df["Source"] = "Simulated Data"
    

    # Plot the first point cloud (UMAP_1)
    plt.figure(figsize=(10, 10))
    plt.scatter(embed_loosened_df["UMAP_1"], embed_loosened_df["UMAP_2"], c='red', label='2lSS1tau', s=1, alpha=0.1)
    plt.scatter(embed_restricted_df["UMAP_1"], embed_restricted_df["UMAP_2"], c='blue', label='2l1tau + all jets', s=1, alpha=1.0)
    plt.scatter(embed_generated_df["UMAP_1"], embed_generated_df["UMAP_2"], c='green', label='2l1tau + all jets generated', s=1, alpha=1.0)
    plt.title(r"Comparison in the data space $\mathcal{X}$", fontsize=36)
    plt.xlabel("UMAP Component 1", fontsize=28)  # Increase label size
    plt.ylabel("UMAP Component 2", fontsize=28)  # Increase label size
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    legend2 = plt.legend(fontsize=28, scatterpoints=20)  # Increase legend size
    legend2.get_frame().set_alpha(0.8)
    plt.grid(True)
    plt.savefig(f'{PATH_RESULTS}vae_data_umap_results.pdf')
    # plt.show()