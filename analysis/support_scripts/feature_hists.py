import ROOT
import pandas as pd

def main():
    PATH_DATA = '/home/lucas/Documents/KYR/msc_thesis/vae-generator-for-particle-physics/analysis/data/'
    PATH_OUT = '/home/lucas/Documents/KYR/msc_thesis/vae-generator-for-particle-physics/analysis/hists/'
    DATA_FILE = 'df_low'
    
    EPOCHS = 100
    
    df = pd.read_csv(f'{PATH_DATA}{DATA_FILE}_disc_{EPOCHS}_new_sym.csv')
    output_file = ROOT.TFile(f"{PATH_OUT}/{DATA_FILE}_disc_{EPOCHS}_new_sym.root", "RECREATE")
    for column_name in df.columns:
        column_array = df[column_name].values
        print(f'Column Name: {column_name}')
        print('NumPy Array:')
        print(column_array)
        print('----------------------')
        min_val = min(column_array)
        max_val = max(column_array)
        num_bins = 50
        margin = 5*(max_val - min_val)/num_bins
        hist_range = (min_val - margin, max_val + margin)
        
        # Create a 1D histogram using ROOT.TH1F
        histogram = ROOT.TH1F(f"{column_name}", f"{column_name} {EPOCHS};X-axis Label;Y-axis Label", num_bins, hist_range[0], hist_range[1])

        # Fill the histogram with data
        for value in column_array:
            histogram.Fill(value)

        # You can also draw the histogram if needed
        #canvas = ROOT.TCanvas("canvas", "Histogram Canvas", 800, 600)
        histogram.Draw()
        
        
        histogram.Write()
    output_file.Close()
        
        
if __name__ == "__main__":
    main()