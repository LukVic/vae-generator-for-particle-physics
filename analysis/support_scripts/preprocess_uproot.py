import uproot
import awkward as ak
import pandas as pd
from root_pandas import to_root
import ROOT

PATH_DATA = 'data/'

file_mc16 = [
    [4083, 4147, 6886, 7012, 9529, 10157, 11710, 15260, 16065, 16208, 17427],
    [2593, 3522, 7859, 12470, 18359, 20366, 21112, 24540, 27476],
    [5476, 7943, 8110, 10038, 10665, 11475, 12169, 15298, 15430, 16586, 26007, 26164, 29389, 29976, 30721, 33568, 33750, 39113]
]

classes = ['tth','ttw','ttz','tt','vv','other']
df_vec_comb = pd.DataFrame()


for k, year in enumerate(['mc16a', 'mc16d', 'mc16e']):
    root_file_path = "/eos/atlas/atlascerngroupdisk/phys-higgs/HSG8/multilepton_ttWttH/v08/v0801_2l1tau/2l1au/nominal/"+ year +"/p4308/410470.root"
    tree_name = "nominal"
    to_remove  = ['jets_pt', 'jets_e', 'jets_eta', 'jets_phi']
    bad_features = ['dilep_type', 'lep_isolationLoose_VarRad_0', 'lep_isolationLoose_VarRad_1', 'taus_JetRNNSigTight_0', 'taus_fromPV_0', 'taus_numTrack_0', 'total_charge', 'nJets_OR']
    with uproot.open(root_file_path) as file:
        my_tree = file[tree_name]
        vector_dict = {f'{feature}_{i}': [] for i in range(len(to_remove)) for feature in to_remove}
        bad_dict = {f'{feature}': [] for feature in bad_features}
        problem_dict = {**vector_dict, **bad_dict}
        print(problem_dict)
        for feature in bad_features:
            print(f"{feature} for {year} -> IN PROCESSING")
            branch_data = my_tree[feature].array()
            problem_dict[feature] = branch_data

        for feature in to_remove:
            print(f"{feature} for {year} -> IN PROCESSING")
            branch_data = my_tree[feature].array()
            
            for entry in branch_data:
                for i in range(4):
                    if i < len(entry):
                        problem_dict[f'{feature}_{i}'].append(entry[i])
                    else:
                        problem_dict[f'{feature}_{i}'].append(0)
        print(problem_dict['dilep_type'][0])
        df_vec = pd.DataFrame(problem_dict)
        #df_vec = df_vec.iloc[file_mc16[k]]
        df_vec_comb = pd.concat([df_vec_comb, df_vec], axis=0)

df_full = pd.DataFrame()
df_no_vec_features = pd.read_csv(PATH_DATA + 'df_'+classes[3]+'_no_vec_features.csv', encoding='ISO-8859-1')
df_vec_comb.to_csv(PATH_DATA + 'df_comb'+classes[3]+'.csv')

df_vec_comb.index = list(range(0,89094))
print(df_no_vec_features)
print(df_vec_comb)
df_full = pd.concat([df_no_vec_features, df_vec_comb],axis=1)


print(df_full)
df_full.to_csv(PATH_DATA + 'df_'+classes[3]+'.csv', index=False)