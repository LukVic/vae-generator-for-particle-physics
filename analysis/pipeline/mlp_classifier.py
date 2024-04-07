import csv
import pandas as pd
import torch
from torch.autograd import grad
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def mlp_classifier(X_train, X_test, y_train, y_test):
    features = X_train.columns.tolist()
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).cuda()
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).cuda()
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.long).cuda()
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.long).cuda()


    input_dim = X_train.shape[1]
    hidden_dim = 512
    output_dim = len(set(y_train))

    # Initialize the model
    model = MLP(input_dim, hidden_dim, output_dim).cuda()

    class_weights = torch.tensor([1.0, 1.0]).cuda()
    # Define loss function and optimizer
    criterion = nn.BCELoss(weight=class_weights)#CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Define data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=2048, shuffle=True)

    # Training loop
    epochs = 30
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.cuda()
            
            outputs = model(inputs)
            outputs = torch.sigmoid(outputs)
            labels_one_hot = F.one_hot(labels, num_classes=2).float().cuda()
            loss_1 = criterion(outputs, labels_one_hot)
            loss_2 = signif_loss(inputs, outputs, labels)
            
            loss = loss_1 #loss_2
            
            optimizer.zero_grad()
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_dataset)
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}')

    
    # Compute gradients of the loss with respect to input features
    inputs = X_train_tensor.requires_grad_(True)
    outputs = model(inputs)
    outputs = torch.sigmoid(outputs)
    labels_one_hot = F.one_hot(y_train_tensor, num_classes=2).float().cuda()
    loss = criterion(outputs, labels_one_hot)
    grads = grad(loss, inputs)[0]

    # Compute feature importance based on gradients
    feature_vals = torch.abs(grads).mean(dim=0)

    # Print feature importance values
    print("Feature Importance:")
    feature_importance(feature_vals.cpu(), features, X_train, scaler)
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        outputs_train = model(X_train_tensor)
        _, predicted_train = torch.max(outputs_train, 1)
        accuracy_train = (predicted_train == y_train_tensor).sum().item() / len(y_train_tensor)
        print(f'Train Accuracy: {accuracy_train:.4f}')
        
        outputs_test = model(X_test_tensor)
        _, predicted_test = torch.max(outputs_test, 1)
        accuracy_test = (predicted_test == y_test_tensor).sum().item() / len(y_test_tensor)
        print(f'Test Accuracy: {accuracy_test:.4f}')
    
    return predicted_train.cpu(), F.softmax(outputs_train, dim=1).cpu(), predicted_test.cpu(), F.softmax(outputs_test, dim=1).cpu()
    

def signif_loss(inputs, outputs, labels):

    y_pred_probs_normalized = outputs / torch.sum(outputs, dim=1, keepdim=True)
    y_pred_labels_5 = (y_pred_probs_normalized[:, 1] > 0.5).long()
    y_pred_labels_3 = (y_pred_probs_normalized[:, 1] > 0.3).long()
    y_pred_labels_9 = (y_pred_probs_normalized[:, 1] > 0.9).long()

    TP = torch.sum((y_pred_labels_5 == 1) & (labels == 1)).float()
    FP = torch.sum((y_pred_labels_5 == 1) & (labels == 0)).float()
    signif_5 = (TP/torch.sqrt(FP + 1e-10)).requires_grad_(True)
    
    TP = torch.sum((y_pred_labels_3 == 1) & (labels == 1)).float()
    FP = torch.sum((y_pred_labels_3 == 1) & (labels == 0)).float()
    signif_3 = (TP/torch.sqrt(FP + 1e-10)).requires_grad_(True)
    
    TP = torch.sum((y_pred_labels_9 == 1) & (labels == 1)).float()
    FP = torch.sum((y_pred_labels_9 == 1) & (labels == 0)).float()
    signif_9 = (TP/torch.sqrt(FP + 1e-10)).requires_grad_(True)
    
    return -torch.log(signif_3 + signif_5 + signif_9)

def feature_importance(vals, features, X_train, scaler):
    n = 10 
    df = pd.DataFrame(scaler.inverse_transform(X_train), columns=features)
    top_features = sorted(zip(features, vals), key=lambda x: x[1], reverse=True)[:n]
    top_feature_names = [feature[0] for feature in top_features]
    top_feature_importance = [feature[1] for feature in top_features]
    df_top_features = pd.DataFrame({'Feature': top_feature_names, 'Importance': top_feature_importance})
    df_best = df[top_feature_names]
    
    plt.figure(figsize=(8, 6))
    plt.bar(top_feature_names, top_feature_importance, color='skyblue')
    plt.xlabel('Feature')
    plt.ylabel('Importance')
    plt.title('Feature Importance')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    #plt.show()
    
    # save_best_features(top_feature_names)
    
    print(df_best.columns)
    
    for column in df_best.columns:
        print(f'{column}: {len(set(df[column]))}')
    

def save_best_features(top_features):
    PATH = '/home/lucas/Documents/KYR/msc_thesis/vae-generator-for-particle-physics/analysis/features/'
    FILE = f'features_top_{len(top_features)}.csv'
    helper_features = ['sig_mass', 'weight', 'row_number', 'file_number']
    all_features = top_features + helper_features
    
    with open(PATH + FILE, 'w', newline='') as file:
        writer = csv.writer(file)
        for feature in all_features:
            writer.writerow([feature])

    print(f"The list has been saved to {PATH+FILE}.")
    

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))  
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x