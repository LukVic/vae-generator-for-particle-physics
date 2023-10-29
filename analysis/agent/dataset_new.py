import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def dataset_regen(PATH_DATA, DATA_FILE, PATH_MODEL, EPOCHS):
    
    print(f'{PATH_MODEL}{DATA_FILE}_{EPOCHS}.pth')
    model = torch.load(f'{PATH_MODEL}{DATA_FILE}_disc_{EPOCHS}.pth')
    df_real = pd.read_csv(f'{PATH_DATA}{DATA_FILE}.csv')
    
    train_dataset = torch.tensor(df_real.values, dtype=torch.float32)
    input_size = train_dataset.shape[1]
    scaler = StandardScaler()
    train_dataset_norm = scaler.fit_transform(train_dataset)
    
    model.eval()
    latent_dimension = 21
    
    data_array = np.empty((0, input_size), dtype=np.float32)

    with torch.no_grad():
        latent_samples = torch.randn(train_dataset.shape[0], latent_dimension)
        x_hats = model.decoder(latent_samples.to('cuda'))
        print(x_hats[:,7])
        #! MAP [0, 1] -> [-1, 1]
        thresholded_eighth_column = torch.where(x_hats[:,7] < 0.5, -1, 1)
        x_hats[:, 7] = thresholded_eighth_column
        x_hats_denorm = scaler.inverse_transform(x_hats.cpu().numpy())
        data_array = np.vstack((data_array, x_hats_denorm))

    # Create a DataFrame from the NumPy array
    df_regen = pd.DataFrame(data_array,columns=df_real.columns)

    print("Processing completed.")

    df_regen.to_csv(f'{PATH_DATA}{DATA_FILE}_disc_{EPOCHS}_new.csv', index=False)

def main():
    PATH_MODEL = '/home/lucas/Documents/KYR/msc_thesis/vae-generator-for-particle-physics/analysis/models/'
    PATH_DATA = '/home/lucas/Documents/KYR/msc_thesis/vae-generator-for-particle-physics/analysis/data/'
    DATA_FILE = 'df_low'
    EPOCHS = 100
    
    dataset_regen(PATH_DATA, DATA_FILE, PATH_MODEL, EPOCHS)
    
if __name__ == "__main__":
    main()