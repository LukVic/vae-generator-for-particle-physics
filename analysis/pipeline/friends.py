import uproot
import pandas as pd
import ROOT

def friend_output(df_output_trex: pd.DataFrame, df_features: pd.DataFrame, conflate: bool):
    file_path = '../data/output_trex/'

    if conflate:
        df_merged = pd.concat([df_features, df_output_trex], axis=1)
        print(df_merged.columns)
        df_merged = df_merged.drop(columns='file_number')
        
        # Convert DataFrame to a dictionary of NumPy arrays
        data_dict = {col: df_merged[col].values for col in df_merged.columns}
        
        # Create RDataFrame from the NumPy dictionary
        rdf = ROOT.RDF.FromNumpy(data_dict)
        
        # Create a ROOT file to save the histograms
        output_file = ROOT.TFile(f'{file_path}tbh_800_new_limit.root', "RECREATE")
        
        for col in df_merged.columns:
            # Define a histogram for each column (you can adjust the binning)
            min_val = df_merged[col].min()
            max_val = df_merged[col].max() + 1/100 * df_merged[col].max()
            hist = rdf.Histo1D((col, col, 100, min_val, max_val), col)
            hist.Write()  # Write the histogram to the ROOT file

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
        
    exit()


    