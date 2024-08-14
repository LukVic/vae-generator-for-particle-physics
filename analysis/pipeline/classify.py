import pandas as pd
import logging
import joblib
import umap
import confmatrix_prettyprint as cm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, auc, roc_curve, precision_score
from aux_functions import *
from mlp_classifier import *
from correlation import *
from optimize import optimize
from dataloader import load_config, load_features
from train import train_model
from friends import friend_output
from aux_functions import num_to_mass, mass_to_num

def classify():
    PATH_JSON = f'../config/' # config path
    gen_params = load_config(PATH_JSON)['classify']['general'] # general parameters for the classification part 
    
    OPTIMIZE = gen_params['optimize'] # use Optuna to finetune hyperparameters
    TYPE = gen_params['type'] # which model should be used
    DEEP = gen_params['deep'] # gbdt or mlp
    AUGMENT = gen_params['augment'] # add artificial data
    ONE_MASS = gen_params['one_mass'] # if only one mass considered for classification
    METRICES = gen_params['metrices']
    
    SPLIT_SEED = gen_params['split_seed'] # array of seeds used for cross validation

    # simulated data fractions
    sim_fractions = gen_params['sim_fractions'] #[0.99] [0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0], [0.2, 0.4, 0.6, 0.8, 0.99]
    # augmented data fractions
    aug_fractions = gen_params['aug_fractions'] #[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99], [0.2, 0.4, 0.6, 0.8, 0.99]
    
    PATH_DATA = '../data/common/'
    PATH_LOGGING = '../logging/'
    FILE_DATA = 'df_all_pres_strict'
    PATH_MODEL =  f'{PATH_DATA}xgboost_model_trained_pres.pkl'
    LOG_FILE_PATH = f'{PATH_LOGGING}results_final.log'
    PATH_FEATURES = f'../features/' # feature directory path
    FEATURES_FILE = f'features_top_10' # 'df_phi', 'df_no_zeros', 'df_8', 'df_pt'  feature file
    
    
    helper_features = gen_params['helper_features'] # features helping to analyse the data
    features_used = load_features(PATH_FEATURES, FEATURES_FILE) # read features from the csv file

    # classes = { 0: 'tbh_all', 1: 'tth', 2: 'ttw', 3: 'ttz', 4: 'tt' }
    classes = { 0: gen_params['sig_mass_label'], 1: 'bkg_all' }
    logg = gen_params["logg"]
    model = None
    
    df_data = pd.read_pickle(f'{PATH_DATA}{FILE_DATA}.pkl')
    

    #df_data['tau_lep_charge_diff'] = df_data['total_charge'] * df_data['taus_charge_0']
    df_data = df_data[features_used + helper_features]
    
    print(set(df_data['sig_mass']))
    # keep only events corresponding to the given mass + background
    if ONE_MASS: df_data = df_data[(df_data['sig_mass'] == 0) | (df_data['sig_mass'] == mass_to_num(gen_params['sig_mass_label']))]

    df_data.loc[df_data['y'] == 0, 'weight'] *= 0.403
    print('WEIGHT SUM:', df_data.loc[df_data['y'] == 0, 'weight'].sum())

    y = df_data['y']
    X = df_data.drop(columns=['y'])
    
    for seed in SPLIT_SEED:
        if logg: 
            with open(LOG_FILE_PATH, 'a') as file: file.write(f'seed_{seed}_params_ = [\n') # write the seed and the opening bracket
        for frac_sim in sim_fractions:
            if frac_sim != 1.0: aug_fractions = [0.0]
            else: aug_fractions = [1.0]#[0.0, 0.2, 0.4, 0.6, 0.8, 0.99]
            for frac_aug in aug_fractions:
                for deep in DEEP:
                    
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=X['class'], random_state=seed)
                    
                    df_test = pd.concat([X_test, y_test], axis=1)
                    df_train = pd.concat([X_train, y_train], axis=1)
                    
                    print(f"SUM OF ALL WEIGHTS: {X['weight'].sum()}")
                    print(f"SUM OF TEST WEIGHTS: {X_test['weight'].sum()}")
                    
                    # just one mass needs to be taken for testing
                    if not ONE_MASS:
                        df_test = df_test[(df_test['sig_mass'] == 0) | (df_test['sig_mass'] == MASS)]
                        y_test = df_test['y']
                        X_test = df_test.drop(columns=['y'])
                    

                    df_train = df_train.sample(frac=frac_sim, random_state=seed) # choose fraction of simulated data
                    
                    weight = X_test['weight']
                    X_train = X_train.drop(columns=list(set(helper_features)-set(['y'])))
                    file_number = X_test['file_number']
                    reaction = X_test['class'] 
                    X_test = X_test.drop(columns=list(set(helper_features)-set(['y'])))
                    
                    # Optuna used to optimize hyperparameters
                    #! just XGB supported at the moment
                    if OPTIMIZE:
                        optimize(X_train, y_train, X_test, y_test, weight)
                        continue
                    
                    #! DATA AUGMENTATION
                    if AUGMENT:
                        events_dict = { 
                        gen_params['sig_mass_label']: X[X['sig_mass'] != 0].shape[0], #10449
                        'bkg_all': X[X['sig_mass'] == 0].shape[0] #27611
                        }
                        df_augment_train = pd.DataFrame()
                        for cl in [gen_params['sig_mass_label'],'bkg_all']:
                            PATH_GEN_MODEL = f'../data/{cl}_input/'
                            df_generated = pd.read_csv(f'{PATH_GEN_MODEL}generated_df_{cl}_pres_strict_E20000_S{events_dict[cl]}_{TYPE}.csv')

                            df_generated = df_generated.sample(frac=frac_aug, random_state=seed)
                            df_augment_train = pd.concat([df_augment_train, df_generated])
                        
                        df_train_all = df_augment_train if gen_params['train_aug_only'] else pd.concat([df_train, df_augment_train])
                        
                        if METRICES:
                            # #wrap_prdc(X_test_strict, df_train)        
                            wasserstein_dist(df_train[df_train['y'] == 0], df_augment_train[df_augment_train['y'] == 0], features_used, cls='SIG')
                            wasserstein_dist(df_train[df_train['y'] == 1], df_augment_train[df_augment_train['y'] == 1], features_used, cls='BKG')
                            # #chi2_test(X_test_strict, df_augment_train, features_used)
                            correlation_matrix(df_train[df_train['y'] == 0], df_augment_train[df_augment_train['y'] == 0], features_used, cls='SIG', TYPE=TYPE)
                            correlation_matrix(df_train[df_train['y'] == 1], df_augment_train[df_augment_train['y'] == 1], features_used, cls='BKG', TYPE=TYPE)
                        
                            X_augment = df_augment_train.drop(columns=['y'])
                            compute_embed(X_train.values, X_augment.values, TYPE, 3000)
                            visualize_embed(TYPE)
                        
                        df_train_all = df_train_all.sample(frac=1.0, random_state=seed) 
                        df_train_all = df_train_all.reset_index(drop=True)
                        
                        y_train = df_train_all['y']
                        X_train = df_train_all.drop(columns=gen_params['helper_features'])

                    
                    print(f"SHAPE OF THE TRAINING DATASET: {X_train.shape}")
                    print(f"SHAPE OF THE TESTING DATASET: {X_test.shape}")
                
                    #! MODEL IS NOT TRAINED AGAIN IF EXISTS
                    # if os.path.exists(PATH_MODEL):
                    #     model = joblib.load(f'{PATH_DATA}xgboost_model_trained_pres.pkl')
                    # else:

                    accuracy_test = 0
                    params = 2
                    
                    if deep:
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
                    

                    output_df = pd.DataFrame(file_number)
                    
                    output_df['y_proba'] = np.array(y_pred_proba_test[:, 0])
                    print(set(reaction))
                    output_df['y'] = reaction
                    output_df['weight'] = weight
                    

                    friend_output(output_df, X_test, gen_params)
                    
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

                    plot_ouput(weight = weight, y_true=y_test, y_probs=y_pred_proba_test[:,0], PATH_RESULTS=PATH_DATA)
                    logging.basicConfig(filename=f'{PATH_LOGGING}results_final.log', level=logging.INFO, format='%(message)s')

                    if logg:
                        with open(LOG_FILE_PATH, 'a') as file:
                            # line_to_write = f'\'DEEP: {deep} | FRS: {frac_sim:.1f} | FRG: {frac_aug:.1f} | PREC TR: {precision_train:.5f} | PREC TE: {precision_test:.5f} | ACC TR: {accuracy_train:.5f} | ACC TE: {accuracy_test:.5f} | SIGT: {best_signif_true:.5f} | SIGS: {best_signif_simp:.5f} | BS: {bs:.5f} | BB: {bb:.5f} | W. RAT: {all_weight_sum/test_weight_sum:.5f} | PARAMS: {params}\',\n'
                            line_to_write = f'\'DEEP: {deep} | FRS: {frac_sim:.1f} | FRG: {frac_aug:.1f} | PREC TR: {precision_train:.5f} | PREC TE: {precision_test:.5f} | ACC TR: {accuracy_train:.5f} | ACC TE: {accuracy_test:.5f} | SIGT: {best_signif_true:.5f} | SIGS: {best_signif_simp:.5f} | BS: {bs:.5f} | BB: {bb:.5f} | PARAMS: {params}\',\n'
                            file.write(line_to_write)
                    print("FINISHED")
            
        with open(LOG_FILE_PATH, 'a') as file:
            file.write(f']\n')



def main():
    classify()

if __name__ == "__main__":
    main()