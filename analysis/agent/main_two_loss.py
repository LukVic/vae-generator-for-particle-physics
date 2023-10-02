import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pandas as pd
from tqdm.auto import tqdm
from sklearn.preprocessing import StandardScaler
from architecture_two_loss import VAE

def main():
    PATH_RESULTS = '/home/lucas/Documents/KYR/msc_thesis/vae-generator-for-particle-physics/analysis/results/'
    PATH_DATA = '/home/lucas/Documents/KYR/msc_thesis/vae-generator-for-particle-physics/analysis/data/'
    # Define hyperparameters
    batch_size = 256
    input_size = [56, 92]
    latent_size = 20
    
    lr = 0.001
    num_epochs = 10
    elbo_history = []

    df_continuous = pd.read_csv(f'{PATH_DATA}df_continuous.csv')
    df_categorical = pd.read_csv(f'{PATH_DATA}df_categorical.csv')
    df_categorical_encoded = pd.get_dummies(df_categorical, columns=df_categorical.columns)
    
    train_dataset_continuous = torch.tensor(df_continuous.values, dtype=torch.float32)
    scaler_continuous = StandardScaler()
    train_dataset_norm_continuous = scaler_continuous.fit_transform(train_dataset_continuous)
    train_dataloader_continuous = DataLoader(train_dataset_norm_continuous, batch_size=batch_size, shuffle=True)
    
    train_dataset_categorical = torch.tensor(df_categorical_encoded.values, dtype=torch.float32)
    scaler_categorical = StandardScaler()
    train_dataset_norm_categorical = scaler_categorical.fit_transform(train_dataset_categorical)
    train_dataloader_categorical = DataLoader(train_dataset_norm_categorical, batch_size=batch_size, shuffle=True)
    
    if torch.cuda.is_available():
        print("CUDA (GPU) is available.")
    else:
        print("CUDA (GPU) is not available.")

    # Create model and optimizer
    model = VAE(latent_size, input_size).to('cuda:0')
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # Train the model

    model.train()
    for epoch in range(num_epochs):
        progress_bar = tqdm(total=len(train_dataloader_continuous))
        for i, (x_con, x_cat) in enumerate(zip(train_dataloader_continuous, train_dataloader_categorical)):
            
            xs = [x_con, x_cat]
            xs = [x.view(-1, input_size[idx]).float() for idx, x in enumerate(xs)]
            optimizer.zero_grad()
            x_hats, pz_con, qz_con, phi_cat = model(xs)
            loss = model.loss_function(xs, x_hats, pz_con, qz_con, phi_cat)
            
            loss.backward()
            optimizer.step()
            elbo_history.append(loss.item())
            
            progress_bar.set_description(f'EPOCH: {epoch+1}/{num_epochs} | LOSS: {loss:.7f}')
            progress_bar.update(1)   
        progress_bar.close()  
    exit()
    # Generate some samples and their reconstructions
    model.eval()
    with torch.no_grad():
        x_sample = next(iter(train_dataloader_continuous))
        x_sample = x_sample.view(-1, input_size)
        x_hat, _, _,_,_ = model(x_sample.float())

    num_samples = 18
    fig, axes = plt.subplots(nrows=6, ncols=6)
    udxs = []
    for i, ax in enumerate(axes.flat):
        if i < num_samples*2:
            row = i / (6)
            print(int(row))
            if int(row) % 2 == 0:
                ax.imshow(x_sample[int(row)*6+(i % 6)].view(1, 72), cmap='gray')
            else:
                ax.imshow(x_hat[int(row-1)*6+(i % 6)].to('cpu').view(1, 72), cmap='gray')
            a = int(row)*8+(i % 8)
            b = int(row-1)*8+(i % 8)
            udxs.append(a)
            udxs.append(b)
        ax.axis('off')
    idxs = sorted(udxs)
    print(idxs)
    plt.savefig(PATH_RESULTS + 'visual.png')
    plt.show()


    # Posterior collapse
    with torch.no_grad():
        kl_divs = []
        for batch_idx, data in enumerate(train_dataloader):
            data = data.to('cpu')
            recon_batch, mu, logvar, qz, std = model(data.float())
            z = qz.rsample()
            pz = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
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
        plt.hist(kl_divs_mean, bins=20)
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