import os
import ROOT
import uproot
import numpy as np
import pandas as pd


def friend_output(df_output_trex: pd.DataFrame, df_features: pd.DataFrame, gen_params, PATH_RESULTS: str):
    if not os.path.exists(f'{PATH_RESULTS}trex/'):
        os.makedirs(f'{PATH_RESULTS}trex/')
    
    if gen_params['output_conflate']:
        df_merged = pd.concat([df_features, df_output_trex], axis=1)
        df_merged = df_merged.drop(columns='file_number')

        # Convert DataFrame to a dictionary of NumPy arrays
        data_dict = {col: df_merged[col].values for col in df_merged.columns}


        # Create a ROOT file to save the tree
        if gen_params['augment']: output_file = ROOT.TFile(f'{PATH_RESULTS}trex/aug_limit.root', "RECREATE")    
        else: output_file = ROOT.TFile(f'{PATH_RESULTS}trex/limit.root', "RECREATE")

        # Create a TTree named "nominal"
        tree = ROOT.TTree("nominal", "nominal")

        # Create branches in the tree for each column in the DataFrame
        branches = {}
        for col in df_merged.columns:
            # Determine the type of each column
            if df_merged[col].dtype == np.float64 or df_merged[col].dtype == np.float32:
                branches[col] = np.zeros(1, dtype=np.float64)
                tree.Branch(col, branches[col], f"{col}/D")
            elif df_merged[col].dtype == np.int32 or df_merged[col].dtype == np.int64:
                branches[col] = np.zeros(1, dtype=np.int32)
                tree.Branch(col, branches[col], f"{col}/I")
            else:
                raise TypeError(f"Unsupported data type for column {col}")

        # Fill the tree
        for i in range(len(df_merged)):
            for col in df_merged.columns:
                branches[col][0] = data_dict[col][i]
            tree.Fill()

        # Write the tree to the ROOT file
        tree.Write()

        # Close the ROOT file
        output_file.Close()
        
    else:
        print(df_output_trex)
        grouped = df_output_trex.groupby('file_number')
        print(grouped)
        # Create individual DataFrames
        individual_dfs = {name: group for name, group in grouped}
        # Print each individual DataFrame
        for name, group_df in individual_dfs.items():
            print(f"\nDataFrame for group {name}:")
            print(group_df)
            # Save the DataFrame to CSV
            #group_df.to_csv(f'{file_path}{name}_friend.csv', index=False)
            
            prob_vals = group_df['probs'].to_numpy()
            branches = {'probs': prob_vals}
            
            # Create a ROOT file and write the tree
            with uproot.create(f'{file_path}{name}_friend.root') as file:
                file["nominal"] = branches
        