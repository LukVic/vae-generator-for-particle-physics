import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm
import json

from architecture import VAE


def main():
    OPTIMIZE = False
    PATH_JSON = '/home/lucas/Documents/KYR/msc_thesis/vae-generator-for-particle-physics/analysis/config/'
    PATH_RESULTS = '/home/lucas/Documents/KYR/msc_thesis/vae-generator-for-particle-physics/analysis/results/'
    PATH_DATA = '/home/lucas/Documents/KYR/msc_thesis/vae-generator-for-particle-physics/analysis/data/'
    
    DATA_FILE = 'df_continuous.csv'
    
    df = pd.read_csv(f'{PATH_DATA}{DATA_FILE}')
    train_dataset = torch.tensor(df.values, dtype=torch.float32)
    
    if torch.cuda.is_available():
        print("CUDA (GPU) is available.")
        device = 'cuda'
    else:
        print("CUDA (GPU) is not available.")
        device = 'cpu'
    
    with open(f"{PATH_JSON}hyperparams.json", 'r') as json_file:
        conf_dict = json.load(json_file)
    
    print(conf_dict)
    # Define hyperparameters
    batch_size = 8196
    latent_size = 50
    lr = 0.0003
    num_epochs = 50
    
    if OPTIMIZE:
        #! HERE IS PERFORMED THE OPTIMIZATION
        pass
    
    input_size = train_dataset.shape[1]
    elbo_history = []
        
    scaler = StandardScaler()
    train_dataset_norm = scaler.fit_transform(train_dataset)
    train_dataloader = DataLoader(train_dataset_norm, batch_size=batch_size, shuffle=True)
    

    # Create model and optimizer
    model = VAE(latent_size, device, input_size, conf_dict)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Train the model
    model.train()
    for epoch in range(num_epochs):
        progress_bar = tqdm(total=len(train_dataloader))
        for i, x in enumerate(train_dataloader):
            x = x.view(-1, input_size)
            optimizer.zero_grad()
            x_hat, pz, qz, = model(x.float())
            loss = model.loss_function(x, x_hat, pz, qz)
            loss.backward()
            optimizer.step()
            elbo_history.append(loss.item())
            progress_bar.set_description(f'EPOCH: {epoch+1}/{num_epochs} | LOSS: {loss:.7f}')
            progress_bar.update(1)
        progress_bar.close()     

    
    # Generate some samples and their reconstructions
    model.eval()
    with torch.no_grad():
        x_sample = next(iter(train_dataloader))
        x_sample = x_sample.view(-1, input_size)
        x_hat, _, _ = model(x_sample.float())
    
    print(f"This is the real sample: {x_sample[0]}")
    print(f"THis is the generated sample: {x_hat[0]}")
    
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


    # Posterior collapse
    with torch.no_grad():
        kl_divs = []
        for batch_idx, data in enumerate(train_dataloader):
            _ , pz, qz = model(data.float())
            kl_div = torch.distributions.kl_divergence(qz, pz)
            kl_divs.append(kl_div)
            break
        
        kl_divs = torch.cat(kl_divs, dim=0)
        kl_divs_mean = kl_divs.mean(dim=0)
        
        # plot histogram of averaged kl divergences for each latent space component
        kl_divs_mean = kl_divs_mean.cpu().numpy()
        plt.figure()
        plt.xlabel('Latent vector component')
        plt.ylabel('Mean KL-divergence accros the batch')
        plt.hist(kl_divs_mean, bins=latent_size)
        #plt.xticks(range(0, 20, 1))
        plt.savefig(PATH_RESULTS + 'z_collapse.png')
        plt.show()
            

    # ELBO graph
    plt.figure()
    plt.plot(elbo_history)
    plt.xlabel('Batch number')
    plt.ylabel('Total Loss')
    plt.title('Total Loss vs. Batch number')
    plt.savefig(PATH_RESULTS + 'elbo.png')
    plt.show()

    print("Number of VAE params: {0}".format(model.count_params()))
    
if __name__ == "__main__":
    main()