import ROOT
import pandas as pd
import numpy as np


def pt_check(path):
    data_original = np.array([])
    data_ganerated = np.array([])
    data_ganerated_sym = np.array([])
    etas = []
    
    df_original = pd.read_csv(f'{path}data/pt_sum.csv')
    df_generated = pd.read_csv(f'{path}data/pt_sum_gen.csv')
    df_generated_sym = pd.read_csv(f'{path}data/pt_sum_gen_sym.csv')
    
    print(df_original.shape)
    print(df_generated.shape)
    for (index_original, row_original), (index_generated, row_generated), (index_generated_sym, row_generated_sym) in zip(df_original.iterrows(), df_generated.iterrows(), df_generated_sym.iterrows()):
        #print(f"Row {index_original}: Original = {row_original}, Generated = {row_generated}")
        etas.append([row_original['lep_Eta_0'], row_original['lep_Eta_1']])
        data_original = np.append(data_original, row_original['HT_lep'])
        data_ganerated = np.append(data_ganerated, row_generated['lep_Pt_0'] + row_generated['lep_Pt_1'])
        data_ganerated_sym = np.append(data_ganerated_sym, row_generated_sym['lep_Pt_0'] + row_generated_sym['lep_Pt_1'])
        
    print(max(data_original))
    print(min(data_original))
    #ht_lep = ROOT.RooRealVar("-", r"-", 30297, 1354239)
    # h_ht_lep_original = ROOT.TH1F("h_ht_lep_original",";ht_lep; events (normalized)", 200, 30297, 1354239)
    # h_ht_lep_generated = ROOT.TH1F("h_ht_lep_generated",";ht_lep; events (normalized)", 200, 30297, 1354239)
    
    h_ht_lep_original = ROOT.TH1F("h_ht_lep_original",";ht_lep; events (normalized)", 100, -100000, 400000)
    h_ht_lep_generated = ROOT.TH1F("h_ht_lep_generated",";ht_lep; events (normalized)", 100, -100000, 400000)
    h_ht_lep_generated_sym = ROOT.TH1F("h_ht_lep_generated_sym",";ht_lep; events (normalized)", 100, -100000, 400000)
    
    #print(etas)
    for event_o, event_g, event_g_s, eta in zip(data_original, data_ganerated, data_ganerated_sym, etas):
        if abs(eta[0]) < 2.4 and abs(eta[1]) < 2.4: 
            h_ht_lep_original.Fill(event_o)
            h_ht_lep_generated.Fill(event_g)
            h_ht_lep_generated_sym.Fill(event_g_s)
    
     # Activate storage of sum of squares of weights for automatic error calculation
    #h_ht_lep_original.Sumw2()


    
    # Set histogram style
    h_ht_lep_original.SetLineColor(ROOT.kGreen)
    h_ht_lep_original.SetMarkerStyle(0)  # Adjust marker style for better visibility

    # Set histogram style
    h_ht_lep_generated.SetLineColor(ROOT.kBlue)
    h_ht_lep_generated.SetMarkerStyle(1)
    
    # Set histogram style
    h_ht_lep_generated_sym.SetLineColor(ROOT.kRed)
    h_ht_lep_generated_sym.SetMarkerStyle(1)
    
    # Draw the histogram with error bars
    h_ht_lep_original.Draw("HIST")  # "E" specifies error bars
    
    # Draw the histogram without error bars
    h_ht_lep_generated.Draw("SAME E")  # "HIST" specifies histogram drawing style without error bars

    # Draw the histogram without error bars
    h_ht_lep_generated_sym.Draw("SAME E")  # "HIST" specifies histogram drawing style without error bars

    # Show the canvas
    ROOT.gPad.Update()
    ROOT.gPad.WaitPrimitive()  
    # canvas = ROOT.TCanvas("canvas", "Comparison of Histograms", 800, 600)

    # # Create TGraphErrors for original histogram
    # graph_original = ROOT.TGraph(h_ht_lep_original)
    # graph_original.SetLineColor(ROOT.kBlue)
    # graph_original.SetMarkerStyle(20)  # Adjust marker style for better visibility
    # graph_original.Draw("A")  # Draw with markers and error bars
    # graph_original.SetTitle(";ht_lep; events (normalized)")

    # # Create TGraphErrors for generated histogram
    # graph_generated = ROOT.TGraphErrors(h_ht_lep_generated)
    # graph_generated.SetLineColor(ROOT.kRed)
    # graph_generated.SetMarkerStyle(20)  # Adjust marker style for better visibility
    # graph_generated.Draw("P SAME")  # Draw with markers and error bars

    # # Add a legend
    # legend = ROOT.TLegend(0.7, 0.7, 0.9, 0.9)
    # legend.AddEntry(graph_original, "Original", "lp")
    # legend.AddEntry(graph_generated, "Generated", "lp")
    # legend.Draw()

    # # Update the canvas
    # canvas.Update()

    # # Keep the application alive
    # input("Press enter to exit.")

    # # Clean up
    # del legend
    # del canvas
        
    fOut = ROOT.TFile(f"{path}hists/let_pt_orig_gen_comparison.root", "RECREATE")
    h_ht_lep_original.Write()
    h_ht_lep_generated.Write()
    fOut.Close()
def main():
    PATH = '/home/lucas/Documents/KYR/msc_thesis/vae-generator-for-particle-physics/analysis/'
    pt_check(PATH)

if __name__ == "__main__":
    main()