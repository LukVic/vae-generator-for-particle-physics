import ROOT
import pandas as pd

def main():
    PATH_DATA = '/home/lucas/Documents/KYR/msc_thesis/vae-generator-for-particle-physics/analysis/data/'
    PATH_OUT = '/home/lucas/Documents/KYR/msc_thesis/vae-generator-for-particle-physics/analysis/hists/'
    DATA_FILE = 'df_low_regen'
    
    df = pd.read_csv(f'{PATH_DATA}{DATA_FILE}.csv')
    for column_name in df.columns:
        column_array = df[column_name].values
        print(f'Column Name: {column_name}')
        print('NumPy Array:')
        print(column_array)
        print('----------------------')
        
        num_bins = 50
        hist_range = (-5.0, 5.0)

        # Create a 1D histogram using ROOT.TH1F
        histogram = ROOT.TH1F("histogram", "Histogram Title;X-axis Label;Y-axis Label", num_bins, hist_range[0], hist_range[1])

        # Fill the histogram with data
        for value in column_array:
            histogram.Fill(value)

        # You can also draw the histogram if needed
        canvas = ROOT.TCanvas("canvas", "Histogram Canvas", 800, 600)
        histogram.Draw()
        
        output_file = ROOT.TFile(f"{PATH_OUT}{DATA_FILE}/{column_name}_regen.root", "RECREATE")
        histogram.Write()
        output_file.Close()
        
        
if __name__ == "__main__":
    main()