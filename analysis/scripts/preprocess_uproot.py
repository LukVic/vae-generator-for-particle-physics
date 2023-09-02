import uproot
import awkward as ak
# Specify the path to the ROOT file
root_file_path = "/eos/atlas/atlascerngroupdisk/phys-higgs/HSG8/multilepton_ttWttH/v08/v0801_2l1tau/2l1au/nominal/mc16a/p4308/410470.root"
tree_name = "nominal;1"

# Open the ROOT file
with uproot.open(root_file_path) as file:
    # List the keys (objects) in the file
    
    my_tree = file[tree_name]
    df = my_tree.pandas.df()
    
    my_tree_keys = my_tree.keys()
    branch_data = my_tree['jets_e'].array()
    numpy_array = ak.to_numpy(branch_data)


    for idx, entry in enumerate(branch_data):
        print(entry)
        if idx == 100:
            exit()
    # Print the keys to see what's inside the file
    #print("Keys in the ROOT file:")
    #for key in keys:
    #    print(type(key))