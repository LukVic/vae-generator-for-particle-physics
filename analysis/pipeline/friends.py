import uproot

def friend_output(df_output_trex):
    file_path = '../data/output_trex/'

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


    