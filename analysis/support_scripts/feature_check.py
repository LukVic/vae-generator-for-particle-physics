import os
import sys
sys.path.append("/home/lucas/Documents/KYR/msc_thesis/vae-generator-for-particle-physics/analysis/pipeline")
sys.path.append("/home/lucas/Documents/KYR/msc_thesis/vae-generator-for-particle-physics/analysis/agent")
import ROOT
import pandas as pd
import numpy as np
import logging
import ctypes
import csv
from scipy.stats import chisquare
from dataloader import load_config, load_features

def feature_check(path):
    
    logging.basicConfig(filename='/home/lucas/Documents/KYR/msc_thesis/vae-generator-for-particle-physics/analysis/logging/chi2_test.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    binning = 100
    
    EPOCHS_STD = 200
    EPOCHS_SYM = 200
    
    #reaction = 'tbh_800_new'
    reaction = 'bkg_all'
    
    # DATASET = 'df_no_zeros'
    # FEATURES = 'low_features'
    # DATASET = 'df_phi'
    # FEATURES = 'phi_features'
    #DATASET = 'df_8'
    #FEATURES = 'features_8'
    # DATASET = 'df_pt'
    # FEATURES = 'pt_features'
    DATASET = f'df_{reaction}_pres_strict'
    PATH_FEATURES = f'{path}features/'
    FEATURES_FILE = 'most_important_gbdt_10_tbh_800_new'  #'features_top_10'
    
    data_original = np.array([])
    data_ganerated = np.array([])
    data_ganerated_sym = np.array([])
    
    ch_test_1 = []
    ch_test_2 = []
    
    #EVENTS = 10449
    EVENTS_1 = 10478
    #EVENTS_1 = 12522
    #EVENTS_2 = 27611
    #EVENTS_1 = 31434
    #EVENTS_1 = 65373
    EVENTS_1 = 46338
    # df_original = pd.read_csv(f'{path}data/tt/{DATASET}.csv')
    # df_generated = pd.read_csv(f'{path}data/tt/{DATASET}_disc_{EPOCHS_STD}_{EPOCHS_STD}_std_h.csv')
    # df_generated_sym = pd.read_csv(f'{path}data/tt/{DATASET}_disc_{EPOCHS_SYM}_{EPOCHS_SYM}_sym_h.csv')
    
    TYPE_1 = 'ddgm'
    #TYPE_1 = 'wgan_gp'
    #TYPE_1 = 'lvae_std'
    #TYPE_2 = 'sym'
    
    df_original = pd.read_csv(f'{path}data/{reaction}_input/{DATASET}.csv')
    df_generated = pd.read_csv(f'{path}data/{reaction}_input/generated_df_{reaction}_pres_strict_E{EPOCHS_STD}_S{EVENTS_1}_{TYPE_1}.csv')
    df_generated_sym = pd.read_csv(f'{path}data/{reaction}_input/generated_df_{reaction}_pres_strict_E{EPOCHS_STD}_S{EVENTS_1}_{TYPE_1}.csv')


    #! adjust df_original
    #features_used = ['taus_pt_0', 'MtLepMet', 'met_met', 'DRll01', 'MLepMet', 'minDeltaR_LJ_0', 'jets_pt_0', 'HT', 'HT_lep', 'total_charge']
    features_used = load_features(PATH_FEATURES, FEATURES_FILE)
    #features_used = ['HT', 'minDeltaR_LJ_0', 'MtLepMet', 'DRll01', 'jets_pt_0', 'met_met', 'taus_pt_0', 'MLepMet', 'HT_lep', 'total_charge']
    df_original = df_original[features_used]
    df_generated = df_generated[features_used]
    df_generated_sym = df_generated_sym[features_used]
    df_original = df_original.drop_duplicates(keep='first')

    print(df_generated.shape)
    print(df_original.shape)
    print(df_generated['taus_charge_0'][-10:])
    # print(df_original)
    # print(df_generated)
    # print(df_generated_sym)
    # exit()

    # print(df_original.shape)
    # print(df_generated.shape)
    # print(df_generated_sym.shape)
    
    feature_list = pd.read_csv(f'{path}features/{FEATURES_FILE}.csv', header=None).to_numpy()
    
    
    chi2_sum_std = 0
    chi2_sum_sym = 0
    
    
    logging.info(f'ELBO EPOCHS NUM: {EPOCHS_STD}')
    logging.info(f'SYMM EPOCHS NUM: {EPOCHS_SYM}')
    
    for feature in feature_list:
        print(f"PROCESSING: {feature}")
        data_original = df_original[feature[0]].values
        data_ganerated = df_generated[feature[0]].values
        data_ganerated_sym = df_generated_sym[feature[0]].values
        
        min_all = np.min([data_original,data_ganerated,data_ganerated_sym])
        max_all = np.max([data_original,data_ganerated,data_ganerated_sym])
        
        min_all += 1/min_all
        max_all += 1/max_all
        
        # min_all = 0
        # max_all = 10
        
        # if feature == 'total_charge':
        #     min_all = -2.5
        #     max_all = 2.5
        # if feature == 'MtLepMet':
        #     print(min_all)
        #     print(max_all)
        #     exit()
        h_feature_original = ROOT.TH1F(f"h_{feature[0]}_original",f";{feature[0]}; events (normalized)",binning, min_all, max_all)
        h_feature_generated_std = ROOT.TH1F(f"h_{feature[0]}_generated",f";{feature[0]}; events (normalized)",binning, min_all, max_all)
        h_feature_generated_sym = ROOT.TH1F(f"h_{feature[0]}_generated_sym",f";{feature[0]}; events (normalized)",binning, min_all, max_all)
        

        # if feature == 'total_charge':
        #     print(set(data_ganerated))
        #     exit()
        event_sum = 0
        for event_o, event_g, event_g_s in zip(data_original, data_ganerated, data_ganerated_sym):
            # print(event_g_s)
            event_sum += event_o
            h_feature_original.Fill(event_o)
            
            # if feature != 'total_charge':
            #     h_feature_generated_std.Fill(event_g)
            #     h_feature_generated_sym.Fill(event_g_s)
            # else:
            #     #print(event_g)
            #     if event_g < 0.1: event_g = -2
            #     else: event_g = 2
            #     if event_g_s < 0.1: event_g_s = -2
            #     else: event_g_s = 2
                
            h_feature_generated_std.Fill(event_g)
            h_feature_generated_sym.Fill(event_g_s)
        # if feature == 'taus_charge_0':
        #     exit()
        # print(event_sum)
        # print(h_feature_original)
        # print(h_feature_original.Integral())
        # print(h_feature_generated_std.Integral())
        # print(h_feature_generated_sym.Integral())
        
        int_original = h_feature_original.Integral()
        int_generated_std = h_feature_generated_std.Integral()
        int_generated_sym = h_feature_generated_sym.Integral()
        
        if h_feature_original.Integral() < 0.001: int_original = 1
        if h_feature_generated_std.Integral() < 0.001: int_generated_std = 1
        if h_feature_generated_sym.Integral() < 0.001: int_generated_sym = 1
        
        
        h_feature_original.Scale(1. / int_original)
        h_feature_original.Write()
        h_feature_generated_std.Scale(1. / int_generated_std)
        h_feature_generated_std.Write()
        h_feature_generated_sym.Scale(1. / int_generated_sym)
        h_feature_generated_sym.Write()
        
        
        c1 = ROOT.TCanvas("c1", "Canvas", 800, 600)
        
        h_feature_original.SetLineColor(ROOT.kBlue)
        h_feature_original.SetMarkerStyle(0)

        h_feature_generated_std.SetLineColor(ROOT.kRed)
        h_feature_generated_std.SetMarkerStyle(0)
        
        h_feature_generated_sym.SetLineColor(ROOT.kGreen)
        h_feature_generated_sym.SetMarkerStyle(0)
        
        max_bin_original = h_feature_original.GetMaximumBin()
        max_bin_generated = h_feature_generated_std.GetMaximumBin()
        max_bin_generated_sym = h_feature_generated_sym.GetMaximumBin()


        #print(f"Max Bin Entries (Original): {h_feature_original.GetBinContent(max_bin_original)}")
        #print(f"Max Bin Entries (Generated): {h_feature_generated_std.GetBinContent(max_bin_generated)}")
        #print(f"Max Bin Entries (Generated Sym): {h_feature_generated_sym.GetBinContent(max_bin_generated_sym)}")
        
        num_original = h_feature_original.GetBinContent(max_bin_original)
        num_generated = h_feature_generated_std.GetBinContent(max_bin_original)
        num_generated_sym = h_feature_generated_sym.GetBinContent(max_bin_original)
        
        
        if num_original > num_generated and num_original > num_generated_sym:
            h_feature_original.Draw("HIST")  
            h_feature_generated_std.Draw("SAME HIST E")  
            # h_feature_generated_sym.Draw("SAME HIST E")  
        elif num_generated > num_original and num_generated > num_generated_sym:
            h_feature_generated_std.Draw("HIST E")  
            h_feature_original.Draw("SAME HIST")  
            # h_feature_generated_sym.Draw("SAME HIST E")  
        elif num_generated_sym > num_original and num_generated_sym > num_generated:
            h_feature_generated_sym.Draw("HIST E")  
            h_feature_original.Draw("SAME HIST")  
            # h_feature_generated_std.Draw("SAME HIST E")  
        else:
            h_feature_original.Draw("HIST")  
            h_feature_generated_std.Draw("SAME HIST E")  
            # h_feature_generated_sym.Draw("SAME HIST E")  
        
        ROOT.gStyle.SetOptStat(0)   
        # if feature != 'total_charge':
            
        #     legend = ROOT.TLegend(0.54, 0.7, 0.98, 0.9)

        #     legend.AddEntry(h_feature_original, "Simulated", "l")
        #     legend.AddEntry(h_feature_generated_std, f"Generated", "l")
        #     #legend.AddEntry(h_feature_generated_std, f"Generated {TYPE_1}", "l")
        #     #legend.AddEntry(h_feature_generated_sym, f"Generated {TYPE_2}", "l")
            
        #     h_feature_original.GetXaxis().SetTitleSize(0.048)
        #     h_feature_original.GetYaxis().SetTitleSize(0.048)
        #     h_feature_generated_std.GetXaxis().SetTitleSize(0.048)
        #     h_feature_generated_std.GetYaxis().SetTitleSize(0.048)
        #     # h_feature_generated_sym.GetXaxis().SetTitleSize(0.048)
        #     # h_feature_generated_sym.GetYaxis().SetTitleSize(0.048)
            
        #     h_feature_original.GetXaxis().SetTitleOffset(0.9) 
        #     h_feature_generated_std.GetXaxis().SetTitleOffset(0.9) 
        #     # h_feature_generated_sym.GetXaxis().SetTitleOffset(0.9) 
            
        #     h_feature_original.GetYaxis().SetTitleOffset(1.0) 
        #     h_feature_generated_std.GetYaxis().SetTitleOffset(1.0) 
        #     # h_feature_generated_sym.GetYaxis().SetTitleOffset(1.0) 
            
        #     # latex = ROOT.TLatex()
        #     # latex.SetTextSize(0.052) 
        #     # latex.SetTextFont(52)
        #     # latex.DrawLatexNDC(0.49, 0.52, "ATLAS Work in Progress")
        # else:
            
        legend = ROOT.TLegend(0.42, 0.6, 0.86, 0.9) 

        
        legend.AddEntry(h_feature_original, "Simulated", "l")
        legend.AddEntry(h_feature_generated_std, f"Generated", "l")
        #legend.AddEntry(h_feature_generated_std, f"Generated {TYPE_1}", "l")
        #legend.AddEntry(h_feature_generated_sym, f"Generated {TYPE_2}", "l")
        
        h_feature_original.GetXaxis().SetTitleSize(0.048)
        h_feature_original.GetYaxis().SetTitleSize(0.048)
        h_feature_generated_std.GetXaxis().SetTitleSize(0.048)
        h_feature_generated_std.GetYaxis().SetTitleSize(0.048)
        # h_feature_generated_sym.GetXaxis().SetTitleSize(0.048)
        # h_feature_generated_sym.GetYaxis().SetTitleSize(0.048)
        
        h_feature_original.GetXaxis().SetTitleOffset(0.9) 
        h_feature_generated_std.GetXaxis().SetTitleOffset(0.9) 
        # h_feature_generated_sym.GetXaxis().SetTitleOffset(0.9) 
        
        h_feature_original.GetYaxis().SetTitleOffset(1.0) 
        h_feature_generated_std.GetYaxis().SetTitleOffset(1.0) 
        # h_feature_generated_sym.GetYaxis().SetTitleOffset(1.0) 
        
        # latex = ROOT.TLatex()
        # latex.SetTextSize(0.052) 
        # latex.SetTextFont(52)
        # latex.DrawLatexNDC(0.43, 0.52, "ATLAS Work in Progress")

        
        legend.SetBorderSize(0)
        legend.SetFillStyle(0)

        legend.Draw("SAME")
        
        chi2_statistic = ctypes.c_double(0.0)
        ndf = ctypes.c_int(0)
        igood = ctypes.c_int(0)
        
        h_feature_original.Chi2TestX(h_feature_generated_std, chi2_statistic, ndf, igood, "P WW")
        if ndf.value == 0: result_std = np.inf
        else: result_std = chi2_statistic.value / ndf.value
        chi2_sum_std += result_std
        
        # if result_std < 1:
        #     print(chi2_statistic.value)
        #     print(ndf.value)
        #     print(igood)
        #     exit()
        
        
        h_feature_original.Chi2TestX(h_feature_generated_sym, chi2_statistic, ndf, igood, "P WW")
        if ndf.value == 0: result_sym = np.inf
        else: result_sym = chi2_statistic.value / ndf.value
        chi2_sum_sym += result_sym
        
        print(f"Chi2 test 1: {feature}: {result_std}")
        print(f"Chi2 test 2: {feature}: {result_sym}")
        ch_test_1.append(result_std)
        ch_test_2.append(result_sym)
        
        logging.info(f'CURRENT FEATURE: {feature}')
        logging.info(f'ELBO VAE: {result_std}')
        logging.info(f'SYMM VAE: {result_sym}')
        
        directory = f'{EPOCHS_STD}_{EPOCHS_SYM}_epochs_histogram_comparison/'

        if not os.path.exists(f'{path}results/feature_plots_comparison/{directory}'):
            os.makedirs(f'{path}results/feature_plots_comparison/{directory}')
        

        c1.SaveAs(f"{path}results/feature_plots_comparison/{directory}vae_std_sym_{feature[0]}_comparison_sim_{TYPE_1}.pdf")
    print(feature_list)
    print(ch_test_1)
    print(ch_test_2)
    logging.info(f'SUMMED CHI2 ELBO: {chi2_sum_std}')
    logging.info(f'SUMMED CHI2 SYMM: {chi2_sum_sym}')

        
            
def main():
    PATH = '/home/lucas/Documents/KYR/msc_thesis/vae-generator-for-particle-physics/analysis/'
    feature_check(PATH)

if __name__ == "__main__":
    main()