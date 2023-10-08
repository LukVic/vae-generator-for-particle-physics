import matplotlib.pyplot as plt
import torch
from sklearn.preprocessing import StandardScaler
import json

PATH_RESULTS = '/home/lucas/Documents/KYR/msc_thesis/vae-generator-for-particle-physics/analysis/results/'

def event_regen(input_size, scaler, train_dataloader, PATH_MODEL):    
    model = torch.load(f'{PATH_MODEL}')
    
    model.eval()
    with torch.no_grad():
        x_sample = next(iter(train_dataloader))
        x_sample = x_sample.view(-1, input_size)
        x_hat, _, _ = model(x_sample.float())
    
    x_sample_denorm = scaler.inverse_transform(x_sample.to('cpu'))[0]
    x_hat_denorm = scaler.inverse_transform(x_hat.to('cpu'))[0]
    print(f"This is the real sample: {x_sample_denorm}")
    print(f"This is the generated sample: {x_hat_denorm}")
    
    num_samples = 18
    _, axes = plt.subplots(nrows=6, ncols=6)
    udxs = []
    for i, ax in enumerate(axes.flat):
        if i < num_samples*2:
            row = i / (6)
            if int(row) % 2 == 0:
                ax.imshow(x_sample[int(row)*6+(i % 6)].view(1, input_size), cmap='gray')
            else:
                ax.imshow(x_hat[int(row-1)*6+(i % 6)].to('cpu').view(1, input_size), cmap='gray')
            a = int(row)*8+(i % 8)
            b = int(row-1)*8+(i % 8)
            udxs.append(a)
            udxs.append(b)
        ax.axis('off')
    plt.savefig(PATH_RESULTS + 'visual.png')
    plt.show()

def pos_collapse(train_dataloader, PATH_MODEL, PATH_JSON):
    with open(f"{PATH_JSON}", 'r') as json_file:
        conf_dict = json.load(json_file)
    
    model = torch.load(f'{PATH_MODEL}')
    gen_params = conf_dict["general"]
    
    # Posterior collapse
    with torch.no_grad():
        kl_divs = []
        for _, data in enumerate(train_dataloader):
            _ , pz, qz = model(data.float())
            kl_div = torch.distributions.kl_divergence(qz, pz)
            kl_divs.append(kl_div)
            #break
        
        kl_divs = torch.cat(kl_divs, dim=0)
        kl_divs_mean = kl_divs.mean(dim=0)
        
        # plot histogram of averaged kl divergences for each latent space component
        kl_divs_mean = kl_divs_mean.cpu().numpy()
        plt.figure()
        plt.xlabel('Latent vector component')
        plt.ylabel('Mean KL-divergence accros the batch')
        plt.hist(kl_divs_mean, bins=gen_params["latent_size"])
        #plt.xticks(range(0, 20, 1))
        plt.savefig(PATH_RESULTS + 'z_collapse.png')
        plt.show()
            
def elbo_plot(elbo_history, PATH_MODEL):
    model = torch.load(f'{PATH_MODEL}')
    
    # ELBO graph
    plt.figure()
    plt.plot(elbo_history)
    plt.xlabel('Batch number')
    plt.ylabel('Total Loss')
    plt.title('Total Loss vs. Batch number')
    plt.savefig(PATH_RESULTS + 'elbo.png')
    plt.show()

    print("Number of VAE params: {0}".format(model.count_params()))