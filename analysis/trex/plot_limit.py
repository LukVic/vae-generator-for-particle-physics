import uproot
import numpy as np
import ROOT
import array
def read_tree(file_path, keys, xs_dict):
    
    # Open the ROOT file
    with uproot.open(file_path) as file:
        # Access the 'stats' tree
        tree = file["stats"]
        # Iterate over all branches (histograms)
        histograms = {}
        for branch_name in tree.keys():
            # Read the histogram data
            histogram = tree[branch_name].array(library="np")
            histograms[branch_name] = histogram
            
            if branch_name in set(keys):
                xs_dict[branch_name].append(histogram.item()) 

    # Now you have all histograms stored in the `histograms` dictionary 
    return xs_dict

def plot_xs_limit(xs_dict, mass_vals):
    
    old_new_indices_arr = [[0,3],[3,7]]
    date_lables = ['old', 'new']
    
    for old_new_indices, date_label in zip(old_new_indices_arr, date_lables):
    
        print(mass_vals)
        
        limit_values = np.array(xs_dict['exp_upperlimit'][old_new_indices[0]:old_new_indices[1]])
        limit_1sigma_low = np.array(xs_dict['exp_upperlimit_minus1'][old_new_indices[0]:old_new_indices[1]])
        limit_1sigma_high = np.array(xs_dict['exp_upperlimit_plus1'][old_new_indices[0]:old_new_indices[1]])
        limit_2sigma_low = np.array(xs_dict['exp_upperlimit_minus2'][old_new_indices[0]:old_new_indices[1]])
        limit_2sigma_high = np.array(xs_dict['exp_upperlimit_plus2'][old_new_indices[0]:old_new_indices[1]])
        
        mass_vals_arr = array.array('d',mass_vals[old_new_indices[0]:old_new_indices[1]])
        limit_values = array.array('d',limit_values)
        
        # limit_2sigma_low_bound = 
        # limit_2sigma_up_bound = 
        
        nullptr = array.array('d',np.zeros(len(mass_vals_arr)))
        
        g_limit = ROOT.TGraph(len(mass_vals_arr),array.array('d',mass_vals_arr), array.array('d',limit_values))
        g_limit.SetLineColor(ROOT.kBlack)
        g_limit.SetLineWidth(2)
        #g_limit.SetMarkerStyle(20)
        g_limit.SetLineStyle(2)
        g_limit.SetTitle("")

        # Create TGraphAsymmErrors for the 1σ band
        g_1sigma = ROOT.TGraphAsymmErrors(len(mass_vals_arr), mass_vals_arr, limit_values,
                                            nullptr, nullptr,
                                            limit_values - limit_1sigma_low, limit_1sigma_high - limit_values)
        g_1sigma.SetFillColor(ROOT.kGreen)
        g_1sigma.SetTitle("")  # Remove title from the graph
        # Create TGraphAsymmErrors for the 2σ band
        g_2sigma = ROOT.TGraphAsymmErrors(len(mass_vals_arr), mass_vals_arr, limit_values,
                                            nullptr, nullptr,
                                            limit_values - limit_2sigma_low, limit_2sigma_high - limit_values)
        g_2sigma.SetFillColor(ROOT.kYellow)
        g_2sigma.SetTitle("")  # Remove title from the graph
        # Draw the plot
        c = ROOT.TCanvas("c", "Cross-Section Limit with Brazilian Bands", 800, 600)
        c.SetLogy()
        c.Clear()  # Clear the canvas to avoid overwriting
        # Draw the 2σ band first (it will be in the background)
        g_2sigma.Draw("A3")

        # Draw the 1σ band on top of the 2σ band
        g_1sigma.Draw("3")

        # Draw the central limit values on top of everything
        g_limit.Draw("PL")

        # Customize the plot
        g_2sigma.GetXaxis().SetTitle(r"m_{H^{\pm}} [GeV]")
        g_2sigma.GetYaxis().SetTitle(r"#sigma (pp #rightarrow tbH^{#pm}) #times BR(H^{#pm} #rightarrow Wh #rightarrow #tau #bar{#tau}) [pb]")
        g_2sigma.GetXaxis().SetLimits(mass_vals_arr[0], mass_vals_arr[-1])
        #g_2sigma.GetYaxis().SetRangeUser(0, 0.1)
        

        # Add a legend
        legend1 = ROOT.TLegend(0.59, 0.69, 0.89, 0.89)
        legend1.SetLineColor(0)
        legend1.AddEntry(g_limit, r"Exp. limit", "lp")
        legend1.AddEntry(g_1sigma, r"Expected #pm1#sigma", "f")
        legend1.AddEntry(g_2sigma, r"Expected #pm2#sigma", "f")
        legend1.Draw()


        # Create a TLatex object for the text box
        latex = ROOT.TLatex()
        latex.SetTextSize(0.045)  # Set the text size
        latex.SetTextAlign(13)   # Align at top-left (1: left, 3: top)
        latex.DrawLatexNDC(0.25, 0.85, r"#sqrt{s} = 13 TeV, 140 fb^{-1}")  # (x, y, text)
        latex.DrawLatexNDC(0.25, 0.78, r"95% C.L.")

        # Draw the canvas
        c.Draw()

        # Save the plot as an image
        c.SaveAs(f"cross_section_limit_{date_label}.pdf")

def main():
    
    # List of keys
    keys = ['exp_upperlimit', 'exp_upperlimit_plus1','exp_upperlimit_plus2', 'exp_upperlimit_minus1', 'exp_upperlimit_minus2']
    # Create a dictionary with empty lists as values
    empty_dict = {key: [] for key in keys}
    masses = ['tbH_250_new', 'tbH_800_new', 'tbH_3000_new', 'tbH_300', 'tbH_800', 'tbH_1500', 'tbH_2000']
    
    
    for mass in masses:
        PATH = f'/afs/cern.ch/user/l/lvicenik/atlas/atlas_env/msc_thesis/trex-fitter/test/output_limit_{mass}/Xjob_{mass}_multinom/Limits/Asymptotics/myLimit.root'
        read_tree(PATH, keys, empty_dict)
    
    mass_vals = [int(mass.split('_')[1]) for mass in masses]
    
    plot_xs_limit(empty_dict,mass_vals) 
if __name__ == "__main__":
    main()    