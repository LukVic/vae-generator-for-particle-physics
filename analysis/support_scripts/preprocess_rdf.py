import ROOT
from ROOT import TFile
import os
import csv
import numpy as np
import pandas as pd
# import uproot

def find_directories_with_file(start_path, file_name):
    result = []
    for root, dirs, files in os.walk(start_path):
        if file_name in files:
            result.append(root)
    return result


def filter_events(df, year, ntuple):
    """
    Reduce initial dataset to only events which shall be used for training
    """
    
    pres = "(custTrigMatch_LooseID_FCLooseIso_DLT ) && ( (dilep_type > 0 ) && ( (lep_ID_0*lep_ID_1)>0) ) && ( ((lep_Pt_0>=10e3) && (lep_Pt_1>=10e3)) && ((fabs(lep_Eta_0)<=2.5) & (fabs(lep_Eta_1)<=2.5))&& ((( abs(lep_ID_0) == 13 ) && ( lep_isMedium_0 )&& ( lep_isolationLoose_VarRad_0 ) && ( passPLIVTight_0 ))|| ((( abs(lep_ID_0) == 11 ) & ( lep_isTightLH_0 )&& ( lep_isolationLoose_VarRad_0 ) && ( passPLIVTight_0 )&& ( lep_ambiguityType_0 == 0 )&& ( lep_chargeIDBDTResult_recalc_rel207_tight_0>0.7 ) )&& ( ((!((!((lep_Mtrktrk_atConvV_CO_0<0.1)&& (lep_Mtrktrk_atConvV_CO_0>=0) & (lep_RadiusCO_0>20)))&& ((lep_Mtrktrk_atPV_CO_0<0.1) & (lep_Mtrktrk_atPV_CO_0>=0)))))&& (!((lep_Mtrktrk_atConvV_CO_0<0.1)&& (lep_Mtrktrk_atConvV_CO_0>=0)&& (lep_RadiusCO_0>20))))))&& ( (( abs(lep_ID_1) == 13 ) && ( lep_isMedium_1 )&& ( lep_isolationLoose_VarRad_1 ) && ( passPLIVTight_1 ) )|| ( (( abs(lep_ID_1) == 11 ) && ( lep_isTightLH_1 )&& ( lep_isolationLoose_VarRad_1 ) && ( passPLIVTight_1 )&& ( lep_ambiguityType_1 == 0 )&& ( lep_chargeIDBDTResult_recalc_rel207_tight_1>0.7 ))&& (((!((!((lep_Mtrktrk_atConvV_CO_1<0.1)&& (lep_Mtrktrk_atConvV_CO_1>=0)&& (lep_RadiusCO_1>20)))&& ((lep_Mtrktrk_atPV_CO_1<0.1)&& (lep_Mtrktrk_atPV_CO_1>=0)))))&& (!((lep_Mtrktrk_atConvV_CO_1<0.1)&& (lep_Mtrktrk_atConvV_CO_1>=0)&& (lep_RadiusCO_1>20)))))) )&& ( nTaus_OR==1 ) && ( nJets_OR_DL1r_85>=1 ) && ( nJets_OR>=4 )&& ( ((dilep_type==2) ) || ( abs(Mll01-91.2e3)>10e3))"
    
    return df.Filter(pres, "l2SS1tau for " + ntuple + " [{0}]".format(year))


def define_variables(df, cols, passing):
    """
    Define the variables which shall be used for training
    """
    data = pd.read_csv('vars_final.csv')
    array_data = list(data.values.squeeze())
    print(len(array_data))
    array_data = array_data + ['row_number']
    numpy_data = rdf.AsNumpy(columns = array_data)
    return pd.DataFrame(numpy_data)


if __name__ == "__main__":
    
    PREPROCESS = 0

    classes = ['tth','ttw','ttz','tt','vv','other']

    tth = ['346343.root', '346344.root', '346345.root']
    ttw = ['700168.root', '700205.root']
    ttz = ['700309.root']
    tt = ['410470.root']
    vv = ['363356.root', '363358.root', '363359.root', '363360.root', '363489.root', '364250.root', '364253.root',
'364254.root', '364255.root', '364283.root', '364284.root', '364285.root', '364286.root', '364287.root']
    other = ['304014.root', '342284.root', '342285.root', '364242.root', '364243.root', '364244.root', '364245.root',
'364246.root', '364247.root', '364248.root', '410080.root', '410081.root', '410397.root', '410398.root',
'410399.root', '410408.root', '410560.root']
    
    bad_features = ['dilep_type', 'lep_isolationLoose_VarRad_0', 'lep_isolationLoose_VarRad_1', 'taus_JetRNNSigTight_0', 'taus_fromPV_0', 'taus_numTrack_0', 'total_charge']

    passing_events = 0
    
    data = pd.read_csv('vars_final.csv')
    array_data = list(data.values.squeeze())
    
    df_combined = pd.DataFrame(columns=array_data)

    
    for year in ['mc16a', 'mc16d', 'mc16e']:
        cls = tt
        PATH = '/eos/atlas/atlascerngroupdisk/phys-higgs/HSG8/multilepton_ttWttH/v08/v0801_2l1tau/2l1au/nominal/' + year
        PATH_DATA = 'data/'
        for ntuple in cls: 
            directories = find_directories_with_file(PATH, ntuple)
            for directory in directories:
                filepath = directory  + '/' + ntuple
                
                myFile = TFile(filepath)
                tree = myFile.Get("nominal")
                branch_list = tree.GetListOfBranches()
                rdf = ROOT.RDataFrame("nominal", filepath)
                
                rdf = rdf.Define("row_number", "rdfentry_")

                if PREPROCESS == 1:
                    rdf = filter_events(rdf, year, ntuple)
                passing_events += rdf.Count().GetValue()
                report = rdf.Report()        
                report.Print()
                
                column_names = rdf.GetColumnNames()

                col_exclude = []
                for name in column_names:
                    column_type = rdf.GetColumnType(name)
                df_cut = define_variables(rdf, col_exclude, passing_events)
                if df_combined is None:
                    df_combined = df_cut
                else:
                    df_combined = pd.concat([df_combined, df_cut])

        
    print("SUM OF THE EVENTS: {0}".format(passing_events))
    df_combined.to_csv(PATH_DATA + 'df_'+classes[3]+'_no_vec_features.csv', index=False)