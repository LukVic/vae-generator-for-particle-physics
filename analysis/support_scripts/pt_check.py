import ROOT
import pandas as pd
import numpy as np

import os

def pt_check(path):
    data_original = np.array([])
    data_ganerated = np.array([])
    data_ganerated_sym = np.array([])
    etas = []
    
    df_original = pd.read_csv(f'{path}data/pt_sum.csv')
    df_generated = pd.read_csv(f'{path}data/pt_sum_gen_std.csv')
    df_generated_sym = pd.read_csv(f'{path}data/pt_sum_gen_sym.csv')
    
    #print(df_original.shape)
    #print(df_generated.shape)
    for (index_original, row_original), (index_generated, row_generated), (index_generated_sym, row_generated_sym) in zip(df_original.iterrows(), df_generated.iterrows(), df_generated_sym.iterrows()):
        #print(f"Row {index_original}: Original = {row_original}, Generated = {row_generated}")
        etas.append([row_original['lep_Eta_0'], row_original['lep_Eta_1']])
        data_original = np.append(data_original, row_original['lep_Pt_0'] + row_original['lep_Pt_1'])
        data_ganerated = np.append(data_ganerated, row_generated['lep_Pt_0'] + row_generated['lep_Pt_1'])
        data_ganerated_sym = np.append(data_ganerated_sym, row_generated_sym['lep_Pt_0'] + row_generated_sym['lep_Pt_1'])
        
    #print(max(data_original))
    #print(min(data_original))
    
    h_ht_lep_original = ROOT.TH1F("h_ht_lep_original",";ht_lep; events (normalized)", 100, 0, 300000)

    
    h_ht_lep_generated = ROOT.TH1F("h_ht_lep_generated",";ht_lep; events (normalized)", 100, 0, 300000)

    
    h_ht_lep_generated_sym = ROOT.TH1F("h_ht_lep_generated_sym",";ht_lep; events (normalized)", 100, 0, 300000)

    
    #print(etas)
    for event_o, event_g, event_g_s, eta in zip(data_original, data_ganerated, data_ganerated_sym, etas):
        if abs(eta[0]) < 2.4 and abs(eta[1]) < 2.4: 
            h_ht_lep_original.Fill(event_o)
            h_ht_lep_generated.Fill(event_g)
            h_ht_lep_generated_sym.Fill(event_g_s)
    
     # Activate storage of sum of squares of weights for automatic error calculation
    #h_ht_lep_original.Sumw2()


    h_ht_lep_original.Scale(1. / h_ht_lep_original.Integral())
    h_ht_lep_original.Write()
    h_ht_lep_generated.Scale(1. / h_ht_lep_generated.Integral())
    h_ht_lep_generated.Write()
    h_ht_lep_generated_sym.Scale(1. / h_ht_lep_generated_sym.Integral())
    h_ht_lep_generated_sym.Write()
    
    
    c1 = ROOT.TCanvas("c1", "Canvas", 800, 600)
    
    # Set histogram style
    h_ht_lep_original.SetLineColor(ROOT.kGreen)
    h_ht_lep_original.SetMarkerStyle(0)  # Adjust marker style for better visibility

    # Set histogram style
    h_ht_lep_generated.SetLineColor(ROOT.kBlue)
    h_ht_lep_generated.SetMarkerStyle(1)
    
    # Set histogram style
    h_ht_lep_generated_sym.SetLineColor(ROOT.kRed)
    h_ht_lep_generated_sym.SetMarkerStyle(1)
    
    
    # Perform the chi-square test
    result1 = h_ht_lep_original.Chi2Test(h_ht_lep_generated, "P WW")
    result2 = h_ht_lep_original.Chi2Test(h_ht_lep_generated_sym, "P WW")
    

    # Output the chi-square value and p-value
    #print("Chi-square value:", result)
    # print("Degrees of freedom:", result[1])
    # print("P-value:", result[2])
    
    
    # Draw the histogram without error bars
    h_ht_lep_generated.Draw("HIST E")  # "HIST" specifies histogram drawing style without error bars

    # Draw the histogram without error bars
    h_ht_lep_generated_sym.Draw("SAME HIST E")  # "HIST" specifies histogram drawing style without error bars
    
    # Draw the histogram with error bars
    h_ht_lep_original.Draw("SAME HIST")  # "E" specifies error bars
    
   


        # Create a legend
    legend = ROOT.TLegend(0.5, 0.6, 0.9, 0.9)  # Define legend position (x1, y1, x2, y2)

    # Add entries to the legend
    legend.AddEntry(h_ht_lep_original, "Original", "l")
    legend.AddEntry(h_ht_lep_generated, "Generated Standard", "l")
    legend.AddEntry(h_ht_lep_generated_sym, "Generated Symmetric", "l")
    
    # Set legend style
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)

    # Draw the legend
    legend.Draw("SAME")


    # # Show the canvas
    # ROOT.gPad.Update()
    # ROOT.gPad.WaitPrimitive()  

        
    # fOut = ROOT.TFile(f"{path}hists/let_pt_orig_gen_comparison.root", "RECREATE")
    # h_ht_lep_original.Write()
    # h_ht_lep_generated.Write()
    # fOut.Close()
    
    directory = f'15000_15000_epochs_histogram_comparison/'

    if not os.path.exists(f'{path}results/feature_plots_comparison/{directory}'):
        os.makedirs(f'{path}results/feature_plots_comparison/{directory}')
        

    c1.SaveAs(f"{path}results/feature_plots_comparison/{directory}vae_std_sym_pt_sum_comparison.pdf")
    
    
    
def main():
    PATH = '/home/lucas/Documents/KYR/msc_thesis/vae-generator-for-particle-physics/analysis/'
    pt_check(PATH)

if __name__ == "__main__":
    main()