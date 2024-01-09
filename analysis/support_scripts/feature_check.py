import os
import ROOT
import pandas as pd
import numpy as np

def feature_check(path):
    data_original = np.array([])
    data_ganerated = np.array([])
    data_ganerated_sym = np.array([])
    
    df_original = pd.read_csv(f'{path}data/df_no_zeros.csv')
    df_generated = pd.read_csv(f'{path}data/df_no_zeros_disc_100_900_std.csv')
    df_generated_sym = pd.read_csv(f'{path}data/df_no_zeros_disc_1000_100_new_sym.csv')

    feature_list = pd.read_csv(f'{path}features/low_features.csv', header=None).to_numpy()

    #print(feature_list[0])
    
    for feature in feature_list:
        # for (index_original, row_original), (index_generated, row_generated), (index_generated_sym, row_generated_sym) in zip(df_original.iterrows(), df_generated.iterrows(), df_generated_sym.iterrows()):
        #     #print(f"Row {index_original}: Original = {row_original}, Generated = {row_generated}")
        #     #print(type(feature[0]))
        #     data_original = np.append(data_original, row_original[feature[0]])
        #     data_ganerated = np.append(data_ganerated, row_generated[feature[0]])
        #     data_ganerated_sym = np.append(data_ganerated_sym, row_generated_sym[feature[0]])
        
        data_original = df_original[feature[0]].values
        data_ganerated = df_generated[feature[0]].values
        data_ganerated_sym = df_generated_sym[feature[0]].values
            
        min_all = np.min([data_original,data_ganerated,data_ganerated_sym])
        max_all = np.max([data_original,data_ganerated,data_ganerated_sym])
        
        # min_all = -100000
        # max_all = 100000
        
        #print(min_all)
        h_feature_original = ROOT.TH1F(f"h_{feature[0]}_original",f";{feature[0]}; events (normalized)", 100, min_all, max_all)
        h_feature_generated = ROOT.TH1F(f"h_{feature[0]}_generated",f";{feature[0]}; events (normalized)", 100, min_all, max_all)
        h_feature_generated_sym = ROOT.TH1F(f"h_{feature[0]}_generated_sym",f";{feature[0]}; events (normalized)", 100, min_all, max_all)
        
        for event_o, event_g, event_g_s in zip(data_original, data_ganerated, data_ganerated_sym):
            #print(event_o)
            h_feature_original.Fill(event_o)
            h_feature_generated.Fill(event_g)
            h_feature_generated_sym.Fill(event_g_s)

        h_feature_original.Scale(1. / h_feature_original.Integral())
        h_feature_original.Write()
        h_feature_generated.Scale(1. / h_feature_generated.Integral())
        h_feature_generated.Write()
        h_feature_generated_sym.Scale(1. / h_feature_generated_sym.Integral())
        h_feature_generated_sym.Write()
        
        
        c1 = ROOT.TCanvas("c1", "Canvas", 800, 600)
        
        # Set histogram style
        h_feature_original.SetLineColor(ROOT.kGreen)
        h_feature_original.SetMarkerStyle(0)  # Adjust marker style for better visibility

        # Set histogram style
        h_feature_generated.SetLineColor(ROOT.kBlue)
        h_feature_generated.SetMarkerStyle(0)
        
        # Set histogram style
        h_feature_generated_sym.SetLineColor(ROOT.kRed)
        h_feature_generated_sym.SetMarkerStyle(0)
        
        # Find the bin with the maximum entries for each histogram
        max_bin_original = h_feature_original.GetMaximumBin()
        max_bin_generated = h_feature_generated.GetMaximumBin()
        max_bin_generated_sym = h_feature_generated_sym.GetMaximumBin()

        # Print the number of entries for the bin with the maximum entries
        print(f"Max Bin Entries (Original): {h_feature_original.GetBinContent(max_bin_original)}")
        print(f"Max Bin Entries (Generated): {h_feature_generated.GetBinContent(max_bin_generated)}")
        print(f"Max Bin Entries (Generated Sym): {h_feature_generated_sym.GetBinContent(max_bin_generated_sym)}")
        
        num_original = h_feature_original.GetBinContent(max_bin_original)
        num_generated = h_feature_generated.GetBinContent(max_bin_generated)
        num_generated_sym = h_feature_generated_sym.GetBinContent(max_bin_generated_sym)
        
        if num_original > num_generated and num_original > num_generated_sym:
            h_feature_original.Draw("HIST")  # "E" specifies error bars
            h_feature_generated.Draw("SAME HIST E")  # "HIST" specifies histogram drawing style without error bars
            h_feature_generated_sym.Draw("SAME HIST E")  # "HIST" specifies histogram drawing style without error bars
        elif num_generated > num_original and num_generated > num_generated_sym:
            h_feature_generated.Draw("HIST E")  # "HIST" specifies histogram drawing style without error bars
            h_feature_original.Draw("SAME HIST")  # "E" specifies error bars
            h_feature_generated_sym.Draw("SAME HIST E")  # "HIST" specifies histogram drawing style without error bars
        elif num_generated_sym > num_original and num_generated_sym > num_generated:
            h_feature_generated_sym.Draw("HIST E")  # "HIST" specifies histogram drawing style without error bars
            h_feature_original.Draw("SAME HIST")  # "E" specifies error bars
            h_feature_generated.Draw("SAME HIST E")  # "HIST" specifies histogram drawing style without error bars
            
            
        
        # Create a legend
        legend = ROOT.TLegend(0.5, 0.6, 0.9, 0.9)  # Define legend position (x1, y1, x2, y2)

        # Add entries to the legend
        legend.AddEntry(h_feature_original, "Original", "l")
        legend.AddEntry(h_feature_generated, "Generated Standard", "l")
        legend.AddEntry(h_feature_generated_sym, "Generated Symmetric", "l")
        
        # Set legend style
        legend.SetBorderSize(0)
        legend.SetFillStyle(0)

        # Draw the legend
        legend.Draw("SAME")
        
        
        directory = f'1000_100_epochs_histogram_comparison/'

        if not os.path.exists(f'{path}results/feature_plots_comparison/{directory}'):
            os.makedirs(f'{path}results/feature_plots_comparison/{directory}')
        

        c1.SaveAs(f"{path}results/feature_plots_comparison/{directory}vae_std_sym_{feature[0]}_comparison.pdf")

            
def main():
    PATH = '/home/lucas/Documents/KYR/msc_thesis/vae-generator-for-particle-physics/analysis/'
    feature_check(PATH)

if __name__ == "__main__":
    main()