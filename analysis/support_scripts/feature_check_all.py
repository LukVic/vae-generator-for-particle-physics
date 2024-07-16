import os
import ROOT
import pandas as pd
import numpy as np
import logging
import ctypes
import csv
from scipy.stats import chisquare

def feature_check(path):

    logging.basicConfig(filename='/home/lucas/Documents/KYR/msc_thesis/vae-generator-for-particle-physics/analysis/logging/chi2_test.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    EPOCHS = 20000
    
    #reaction = 'bkg_all'
    reaction = 'tbh_800_new'
    
    DATASET = f'df_{reaction}_pres_strict'
    FEATURES = 'features_top_10'
    
    data_original = np.array([])
    data_ganerated_std = np.array([])
    data_ganerated_sym = np.array([])
    data_ganerated_std_h = np.array([])
    data_ganerated_sym_h = np.array([])
    
    ch_test_std = []
    ch_test_sym = []
    ch_test_std_h = []
    ch_test_sym_h = []
    
    EVENTS = 10449
    #EVENTS = 27611
    # df_original = pd.read_csv(f'{path}data/tt/{DATASET}.csv')
    # df_generated = pd.read_csv(f'{path}data/tt/{DATASET}_disc_{EPOCHS_STD}_{EPOCHS_STD}_std_h.csv')
    # df_generated_sym = pd.read_csv(f'{path}data/tt/{DATASET}_disc_{EPOCHS_SYM}_{EPOCHS_SYM}_sym_h.csv')
    
    TYPE_1 = 'std'
    TYPE_2 = 'sym'
    TYPE_3 = 'std_h'
    TYPE_4 = 'sym_h'
    
    df_original = pd.read_csv(f'{path}data/{reaction}_input/{DATASET}.csv')
    df_generated_std = pd.read_csv(f'{path}data/{reaction}_input/generated_df_{reaction}_pres_strict_E{45000}_S{EVENTS}_{TYPE_1}.csv')
    df_generated_sym = pd.read_csv(f'{path}data/{reaction}_input/generated_df_{reaction}_pres_strict_E{40000}_S{EVENTS}_{TYPE_2}.csv')
    df_generated_std_h = pd.read_csv(f'{path}data/{reaction}_input/generated_df_{reaction}_pres_strict_E{45000}_S{EVENTS}_{TYPE_3}.csv')
    #df_generated_sym_h = pd.read_csv(f'{path}data/{reaction}_input/generated_df_{reaction}_pres_strict_E{EPOCHS}_S{EVENTS}_{TYPE_4}.csv')

    print(df_original.shape)
    print(df_generated_std.shape)
    print(df_generated_sym.shape)
    print(df_generated_std_h.shape)
    #print(df_generated_sym_h.shape)
    
    print(df_generated_std)
    feature_list = pd.read_csv(f'{path}features/{FEATURES}.csv', header=None).to_numpy()
    
    
    chi2_sum_std = 0
    chi2_sum_sym = 0
    chi2_sum_std_h = 0
    chi2_sum_sym_h = 0
    
    
    logging.info(f'EPOCHS NUM: {EPOCHS}')
    
    for feature in feature_list:
        data_original = df_original[feature[0]].values
        data_ganerated_std = df_generated_std[feature[0]].values
        data_ganerated_sym = df_generated_sym[feature[0]].values
        data_ganerated_std_h = df_generated_std_h[feature[0]].values
        #data_ganerated_sym_h = df_generated_sym_h[feature[0]].values
        
        
        min_all = np.min([data_original,data_ganerated_std,data_ganerated_sym,data_ganerated_std_h])
        max_all = np.max([data_original,data_ganerated_std,data_ganerated_sym,data_ganerated_std_h])
        
        if feature == 'total_charge':
            min_all = -2.5
            max_all = 2.5

        h_feature_original = ROOT.TH1F(f"h_{feature[0]}_original",f";{feature[0]}; events (normalized)",50, min_all, max_all)
        h_feature_generated_std = ROOT.TH1F(f"h_{feature[0]}_generated",f";{feature[0]}; events (normalized)",50, min_all, max_all)
        h_feature_generated_sym = ROOT.TH1F(f"h_{feature[0]}_generated_sym",f";{feature[0]}; events (normalized)",50, min_all, max_all)
        h_feature_generated_std_h = ROOT.TH1F(f"h_{feature[0]}_generated",f";{feature[0]}; events (normalized)",50, min_all, max_all)
        #h_feature_generated_sym_h = ROOT.TH1F(f"h_{feature[0]}_generated",f";{feature[0]}; events (normalized)",100, min_all, max_all)
                
        for event_o, event_g_std, event_g_sym, event_g_std_h in zip(data_original, data_ganerated_std, data_ganerated_sym, data_ganerated_std_h):
            h_feature_original.Fill(event_o)
            
            if feature != 'total_charge':
                h_feature_generated_std.Fill(event_g_std)
                h_feature_generated_sym.Fill(event_g_sym)
                h_feature_generated_std_h.Fill(event_g_std_h)
                #h_feature_generated_sym_h.Fill(event_g_sym_h)
            else:
                if event_g_std < 0.1: event_g_std = -2
                else: event_g_std = 2
                if event_g_sym < 0.1: event_g_sym = -2
                else: event_g_sym = 2
                if event_g_std_h < 0.1: event_g_std_h = -2
                else: event_g_std_h = 2
                # if event_g_sym_h < 0.1: event_g_sym_h = -2
                # else: event_g_sym_h = 2
                
                h_feature_generated_std.Fill(event_g_std)
                h_feature_generated_sym.Fill(event_g_sym)
                h_feature_generated_std_h.Fill(event_g_std_h)
                #h_feature_generated_sym_h.Fill(event_g_sym_h)

        h_feature_original.Scale(1. / h_feature_original.Integral())
        h_feature_original.Write()
        h_feature_generated_std.Scale(1. / h_feature_generated_std.Integral())
        h_feature_generated_std.Write()
        h_feature_generated_sym.Scale(1. / h_feature_generated_sym.Integral())
        h_feature_generated_sym.Write()
        h_feature_generated_std_h.Scale(1. / h_feature_generated_std_h.Integral())
        h_feature_generated_std_h.Write()
        # h_feature_generated_sym_h.Scale(1. / h_feature_generated_sym_h.Integral())
        # h_feature_generated_sym_h.Write()
        
        c1 = ROOT.TCanvas("c1", "Canvas", 800, 600)
        
        h_feature_original.SetLineColor(ROOT.kBlack)
        h_feature_original.SetMarkerStyle(0)
        h_feature_generated_std.SetLineColor(ROOT.kBlue)
        h_feature_generated_std.SetMarkerStyle(0)
        h_feature_generated_sym.SetLineColor(ROOT.kRed)
        h_feature_generated_sym.SetMarkerStyle(0)
        h_feature_generated_std_h.SetLineColor(ROOT.kViolet)
        h_feature_generated_std_h.SetMarkerStyle(0)
        # h_feature_generated_sym_h.SetLineColor(ROOT.kOrange)
        # h_feature_generated_sym_h.SetMarkerStyle(0)
        
        
        max_bin_original = h_feature_original.GetMaximumBin()
        max_bin_generated_std = h_feature_generated_std.GetMaximumBin()
        max_bin_generated_sym = h_feature_generated_sym.GetMaximumBin()
        max_bin_generated_std_h = h_feature_generated_std_h.GetMaximumBin()
        # max_bin_generated_sym_h = h_feature_generated_sym_h.GetMaximumBin()

        print(f"Max Bin Entries (Original): {h_feature_original.GetBinContent(max_bin_original)}")
        print(f"Max Bin Entries (Generated ELBO Standard): {h_feature_generated_std.GetBinContent(max_bin_generated_std)}")
        print(f"Max Bin Entries (Generated SEL Standard): {h_feature_generated_sym.GetBinContent(max_bin_generated_sym)}")
        print(f"Max Bin Entries (Generated ELBO Ladder): {h_feature_generated_std_h.GetBinContent(max_bin_generated_std_h)}")
        # print(f"Max Bin Entries (Generated SEL Ladder): {h_feature_generated_sym_h.GetBinContent(max_bin_generated_sym_h)}")
        
        num_original = h_feature_original.GetBinContent(max_bin_original)
        num_generated_std = h_feature_generated_std.GetBinContent(max_bin_generated_std)
        num_generated_sym = h_feature_generated_sym.GetBinContent(max_bin_generated_sym)
        num_generated_std_h = h_feature_generated_std.GetBinContent(max_bin_generated_std_h)
        # num_generated_sym_h = h_feature_generated_sym_h.GetBinContent(max_bin_generated_sym_h)
        

        h_feature_original.Draw("HIST")
        h_feature_generated_std.Draw("SAME HIST E") 
        h_feature_generated_sym.Draw("SAME HIST E")  
        h_feature_generated_std_h.Draw("SAME HIST E")
        #h_feature_generated_sym_h.Draw("SAME HIST E")
        
        
        ROOT.gStyle.SetOptStat(0) 

        if feature == 'DRll01': 
            legend = ROOT.TLegend(0.54, 0.6, 0.98, 0.9)

            legend.AddEntry(h_feature_original, "Simulated", "l")
            legend.AddEntry(h_feature_generated_std, f"Gen. ELBO Standard", "l")
            legend.AddEntry(h_feature_generated_sym, f"Gen. SEL Standard", "l")
            legend.AddEntry(h_feature_generated_std_h, f"Gen. ELBO Ladder", "l")
            # legend.AddEntry(h_feature_generated_sym_h, f"Generated SEL Ladder", "l")
            
            h_feature_original.GetXaxis().SetTitleSize(0.048)
            h_feature_original.GetYaxis().SetTitleSize(0.048)
            h_feature_generated_std.GetXaxis().SetTitleSize(0.048)
            h_feature_generated_std.GetYaxis().SetTitleSize(0.048)
            h_feature_generated_sym.GetXaxis().SetTitleSize(0.048)
            h_feature_generated_sym.GetYaxis().SetTitleSize(0.048)
            h_feature_generated_std_h.GetXaxis().SetTitleSize(0.048)
            h_feature_generated_std_h.GetYaxis().SetTitleSize(0.048)
            # h_feature_generated_sym_h.GetXaxis().SetTitleSize(0.048)
            # h_feature_generated_sym_h.GetYaxis().SetTitleSize(0.048)
            
            
            h_feature_original.GetXaxis().SetTitleOffset(0.9) 
            h_feature_generated_std.GetXaxis().SetTitleOffset(0.9) 
            h_feature_generated_sym.GetXaxis().SetTitleOffset(0.9)
            h_feature_generated_std_h.GetXaxis().SetTitleOffset(0.9)
            # h_feature_generated_sym_h.GetXaxis().SetTitleOffset(0.9) 
            
            h_feature_original.GetYaxis().SetTitleOffset(1.0) 
            h_feature_generated_std.GetYaxis().SetTitleOffset(1.0) 
            h_feature_generated_sym.GetYaxis().SetTitleOffset(1.0)
            h_feature_generated_std_h.GetYaxis().SetTitleOffset(1.0)
            # h_feature_generated_sym_h.GetYaxis().SetTitleOffset(1.0) 
        elif feature != 'total_charge' and feature != 'DRll01' :
            
            legend = ROOT.TLegend(0.52, 0.6, 0.98, 0.9)

            legend.AddEntry(h_feature_original, "Simulated", "l")
            legend.AddEntry(h_feature_generated_std, f"Gen. ELBO Standard", "l")
            legend.AddEntry(h_feature_generated_sym, f"Gen. SEL Standard", "l")
            legend.AddEntry(h_feature_generated_std_h, f"Gen. ELBO Ladder", "l")
            #legend.AddEntry(h_feature_generated_sym_h, f"Gen. SEL Ladder", "l")
            
            h_feature_original.GetXaxis().SetTitleSize(0.048)
            h_feature_original.GetYaxis().SetTitleSize(0.048)
            h_feature_generated_std.GetXaxis().SetTitleSize(0.048)
            h_feature_generated_std.GetYaxis().SetTitleSize(0.048)
            h_feature_generated_sym.GetXaxis().SetTitleSize(0.048)
            h_feature_generated_sym.GetYaxis().SetTitleSize(0.048)
            h_feature_generated_std_h.GetXaxis().SetTitleSize(0.048)
            h_feature_generated_std_h.GetYaxis().SetTitleSize(0.048)
            # h_feature_generated_sym_h.GetXaxis().SetTitleSize(0.048)
            # h_feature_generated_sym_h.GetYaxis().SetTitleSize(0.048)
            
            
            h_feature_original.GetXaxis().SetTitleOffset(0.9) 
            h_feature_generated_std.GetXaxis().SetTitleOffset(0.9) 
            h_feature_generated_sym.GetXaxis().SetTitleOffset(0.9)
            h_feature_generated_std_h.GetXaxis().SetTitleOffset(0.9)
            # h_feature_generated_sym_h.GetXaxis().SetTitleOffset(0.9) 
            
            h_feature_original.GetYaxis().SetTitleOffset(1.0) 
            h_feature_generated_std.GetYaxis().SetTitleOffset(1.0) 
            h_feature_generated_sym.GetYaxis().SetTitleOffset(1.0)
            h_feature_generated_std_h.GetYaxis().SetTitleOffset(1.0)
            # h_feature_generated_sym_h.GetYaxis().SetTitleOffset(1.0) 
        else:
            legend = ROOT.TLegend(0.42, 0.6, 0.86, 0.9) 

            legend.AddEntry(h_feature_original, "Simulated", "l")
            legend.AddEntry(h_feature_generated_std, f"Gen. ELBO Standard", "l")
            legend.AddEntry(h_feature_generated_sym, f"Gen. SEL Standard", "l")
            legend.AddEntry(h_feature_generated_std_h, f"Gen. ELBO Ladder", "l")
            # legend.AddEntry(h_feature_generated_sym_h, f"Generated SEL Ladder", "l")
            
            h_feature_original.GetXaxis().SetTitleSize(0.048)
            h_feature_original.GetYaxis().SetTitleSize(0.048)
            h_feature_generated_std.GetXaxis().SetTitleSize(0.048)
            h_feature_generated_std.GetYaxis().SetTitleSize(0.048)
            h_feature_generated_sym.GetXaxis().SetTitleSize(0.048)
            h_feature_generated_sym.GetYaxis().SetTitleSize(0.048)
            h_feature_generated_std_h.GetXaxis().SetTitleSize(0.048)
            h_feature_generated_std_h.GetYaxis().SetTitleSize(0.048)
            # h_feature_generated_sym_h.GetXaxis().SetTitleSize(0.048)
            # h_feature_generated_sym_h.GetYaxis().SetTitleSize(0.048)
            
            
            h_feature_original.GetXaxis().SetTitleOffset(0.9) 
            h_feature_generated_std.GetXaxis().SetTitleOffset(0.9) 
            h_feature_generated_sym.GetXaxis().SetTitleOffset(0.9)
            h_feature_generated_std_h.GetXaxis().SetTitleOffset(0.9)
            # h_feature_generated_sym_h.GetXaxis().SetTitleOffset(0.9) 
            
            h_feature_original.GetYaxis().SetTitleOffset(1.0) 
            h_feature_generated_std.GetYaxis().SetTitleOffset(1.0) 
            h_feature_generated_sym.GetYaxis().SetTitleOffset(1.0)
            h_feature_generated_std_h.GetYaxis().SetTitleOffset(1.0)
            # h_feature_generated_sym_h.GetYaxis().SetTitleOffset(1.0) 

        
        legend.SetBorderSize(0)
        legend.SetFillStyle(0)

        legend.Draw("SAME")
        
        chi2_statistic = ctypes.c_double(0.0)
        ndf = ctypes.c_int(0)
        igood = ctypes.c_int(0)
        
        
        h_feature_original.Chi2TestX(h_feature_generated_std, chi2_statistic, ndf, igood, "P WW")
        result_std = chi2_statistic.value / ndf.value
        chi2_sum_std += result_std
        
        h_feature_original.Chi2TestX(h_feature_generated_sym, chi2_statistic, ndf, igood, "P WW")
        result_sym = chi2_statistic.value / ndf.value
        chi2_sum_sym += result_sym
        
        h_feature_original.Chi2TestX(h_feature_generated_std_h, chi2_statistic, ndf, igood, "P WW")
        result_std_h = chi2_statistic.value / ndf.value
        chi2_sum_std_h += result_std_h
        
        # h_feature_original.Chi2TestX(h_feature_generated_sym_h, chi2_statistic, ndf, igood, "P WW")
        # result_sym_h = chi2_statistic.value / ndf.value
        # chi2_sum_sym_h += result_sym_h
        
        
        print(f"Chi2 test {feature}: {result_std}")
        ch_test_std.append(round(result_std,3))
        
        print(f"Chi2 test {feature}: {result_sym}")
        ch_test_sym.append(round(result_sym,3))
        
        print(f"Chi2 test {feature}: {result_std_h}")
        ch_test_std_h.append(round(result_std_h,3))
        
        # print(f"Chi2 test {feature}: {result_sym_h}")
        # ch_test_sym_h.append(result_sym_h)
        
        # logging.info(f'CURRENT FEATURE: {feature}')
        # logging.info(f'ELBO VAE: {result_std}')
        # logging.info(f'SYMM VAE: {result_sym}')
        
        directory = f'{EPOCHS}_epochs_histogram_comparison/'

        if not os.path.exists(f'{path}results/feature_plots_comparison/{directory}'):
            os.makedirs(f'{path}results/feature_plots_comparison/{directory}')
        

        c1.SaveAs(f"{path}results/feature_plots_comparison/{directory}vae_{feature[0]}_comparison_{TYPE_1}_{TYPE_2}_{TYPE_3}_{TYPE_4}_{reaction}.pdf")

    print(ch_test_std)
    print(ch_test_sym)
    print(ch_test_std_h)
    # print(ch_test_sym_h)
        
            
def main():
    PATH = '/home/lucas/Documents/KYR/msc_thesis/vae-generator-for-particle-physics/analysis/'
    feature_check(PATH)

if __name__ == "__main__":
    main()
