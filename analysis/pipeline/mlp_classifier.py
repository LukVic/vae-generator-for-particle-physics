import csv
import pandas as pd
import numpy as np
import torch
from torch.autograd import grad
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc



def mlp_classifier(X_train, X_test, y_train, y_test, frac_sim, frac_gen, weights):
    torch.manual_seed(0)
    
    features = X_train.columns.tolist()
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).cuda()
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).cuda()
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.long).cuda()
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.long).cuda()


    input_dim = X_train.shape[1]
    hidden_dim = 2048
    output_dim = len(set(y_train))
    lr = 1e-04

    model = MLP(input_dim, hidden_dim, output_dim).cuda()

    #class_weights = torch.tensor([1.0, 1.0]).cuda()
    criterion = nn.BCELoss()#CrossEntropyLoss()
    
    params = count_parameters(model)

    print(f'NUMBER OF PARAMETERS: {params}')
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_val_loss = np.inf
    epochs_without_improvement = 0
    patience = 30
    eps = 1e-3
    
    best_model_params = None
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=2048*3, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2048*3, shuffle=False)

    trn_loss_history = []
    val_loss_history = []

    trn_acc_history = []
    val_acc_history = []

    trn_pre_history = []    
    val_pre_history = []

    trn_significance_history = []  
    val_significance_history = [] 

    best_accuracy = 0

    epochs = 1000
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        class_correct = [0] * 2  
        class_predicted = [0] * 2  
        false_positives = [0] * 2 

        for inputs, labels in train_loader:
            inputs = inputs.cuda()

            outputs = model(inputs)
            outputs = torch.sigmoid(outputs)
            labels_one_hot = F.one_hot(labels, num_classes=2).float().cuda()
            loss_1 = criterion(outputs, labels_one_hot)
            loss_2 = signif_loss(inputs, outputs, labels)

            loss = loss_1  # loss_2

            optimizer.zero_grad()

            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

            _, predicted = torch.max(outputs, 1)
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)

            for i in range(2):
                class_correct[i] += ((predicted == i) & (labels == i)).sum().item()
                class_predicted[i] += (predicted == i).sum().item()
                false_positives[i] += ((predicted == i) & (labels != i)).sum().item()

        precision = [class_correct[i] / (class_predicted[i] + 1e-9) for i in range(2)]
        overall_precision = sum(class_correct) / sum(class_predicted)
        trn_pre_history.append(precision[1])

        trn_loss = running_loss / len(train_dataset)
        trn_loss_history.append(trn_loss)

        accuracy_train = correct_train / total_train
        trn_acc_history.append(accuracy_train)

        significance_train = np.sqrt(2*(class_predicted[1])*np.log2(1 + class_correct[1]/false_positives[1]) - 2*class_correct[1])
        trn_significance_history.append(significance_train)

        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0

        class_correct = [0] * 2  
        class_predicted = [0] * 2  
        false_positives = [0] * 2 

        with torch.no_grad():
            for inputs_val, labels_val in val_loader:
                inputs_val = inputs_val.cuda()
                labels_one_hot_val = F.one_hot(labels_val, num_classes=2).float().cuda()

                outputs_val = model(inputs_val)
                outputs_val = torch.sigmoid(outputs_val)

                val_loss_1 = criterion(outputs_val, labels_one_hot_val)
                val_loss = val_loss_1 

                running_val_loss += val_loss.item() * inputs_val.size(0)

                _, predicted = torch.max(outputs_val, 1)
                correct_val += (predicted == labels_val).sum().item()
                total_val += labels_val.size(0)

                for i in range(2):
                    class_correct[i] += ((predicted == i) & (labels_val == i)).sum().item()
                    class_predicted[i] += (predicted == i).sum().item()
                    false_positives[i] += ((predicted == i) & (labels_val != i)).sum().item()

        val_loss = running_val_loss / len(val_dataset)
        val_loss_history.append(val_loss)

        accuracy_val = correct_val / total_val
        val_acc_history.append(accuracy_val)

        precision = [class_correct[i] / (class_predicted[i] + 1e-9) for i in range(2)]
        val_pre_history.append(precision[1])

        significance_val = np.sqrt(2*(class_predicted[1])*np.log2(1 + class_correct[1]/false_positives[1]) - 2*class_correct[1])
        val_significance_history.append(significance_val)
        
        if val_loss + eps < best_val_loss:
            print("SAVING NEW BEST MODEL")
            best_val_loss = val_loss
            best_model_params = model.state_dict()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience and epoch > 20:
                print(f'Early stopping at epoch {epoch+1}')
                break
        
        print(f'Epoch [{epoch+1}/{epochs}], Loss TRN: {trn_loss:.4f} | Loss VAL: {val_loss:.4f}')
    
        if epoch % 10 == 0:
            plot_loss(trn_loss_history,val_loss_history, trn_acc_history,val_acc_history, trn_pre_history, val_pre_history, trn_significance_history, val_significance_history, frac_sim, frac_gen, lr, epochs, params)
            plt.close()
            
    inputs = X_train_tensor.requires_grad_(True)
    outputs = model(inputs)
    outputs = torch.sigmoid(outputs)
    labels_one_hot = F.one_hot(y_train_tensor, num_classes=2).float().cuda()
    loss = criterion(outputs, labels_one_hot)
    grads = grad(loss, inputs)[0]
    
    

    feature_vals = torch.abs(grads).mean(dim=0)

    print("Feature Importance:")
    feature_importance(feature_vals.cpu(), features, X_train, scaler)
    model.load_state_dict(best_model_params)

    model.eval()
    with torch.no_grad():
        #print(torch.cuda.memory_summary(device='cuda', abbreviated=False))
        
        outputs_train = model(X_train_tensor)
        _, predicted_train = torch.max(outputs_train, 1)
        accuracy_train = (predicted_train == y_train_tensor).sum().item() / len(y_train_tensor)
        print(f'Train Accuracy: {accuracy_train:.4f}')
        
        outputs_test = model(X_test_tensor)
        _, predicted_test = torch.max(outputs_test, 1)
        accuracy_test = (predicted_test == y_test_tensor).sum().item() / len(y_test_tensor)
        print(f'Test Accuracy: {accuracy_test:.4f}')
    
    return predicted_train.cpu(), F.softmax(outputs_train, dim=1).cpu(), predicted_test.cpu(), F.softmax(outputs_test, dim=1).cpu(), best_accuracy, params

def count_parameters(model):
    # for p in model.parameters():
    #     print(p.numel())
    return sum(p.numel() for p in model.parameters() if p.requires_grad) 

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


def plot_loss(loss_vals_trn, loss_vals_val, acc_vals_trn, acc_vals_val, pre_vals_trn, pre_vals_val, sig_vals_trn, sig_vals_val, frac_sim, frac_gen, lr, epochs, params):
    PATH_SAVE = '/home/lucas/Documents/KYR/msc_thesis/vae-generator-for-particle-physics/analysis/results/mlp_output/final_exp_gg_ss_gs_overfit/'
    plt.clf()
    plt.figure(figsize=(8, 8))

    plt.subplot(2, 2, 1)
    plt.plot(loss_vals_trn, label='Training Loss', color='blue')
    plt.plot(loss_vals_val, label='Validation Loss', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(acc_vals_trn, label='Training Accuracy', color='green')
    plt.plot(acc_vals_val, label='Validation Accuracy', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(pre_vals_trn, label='Training Precision', color='purple')
    plt.plot(pre_vals_val, label='Validation Precision', color='brown')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.title('Training and Validation Precision')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.plot(sig_vals_trn, label='Training Significance', color='cyan')
    plt.plot(sig_vals_val, label='Validation Significance', color='magenta')
    plt.xlabel('Epoch')
    plt.ylabel('Significance')
    plt.title('Training and Validation Significance')
    plt.legend()
    plt.grid(True)

    plt.tight_layout() 

    plt.savefig(f'{PATH_SAVE}final_exp_gg_ss_gs_overfit_{frac_sim}_{frac_gen}_{epochs}_{lr}_{params}_gs.pdf')
    

def feature_importance(vals, features, X_train, scaler):
    n = 30 
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
    
    save_best_features(top_feature_names)
    
    print(df_best.columns)
    
    for column in df_best.columns:
        print(f'{column}: {len(set(df[column]))}')
    

def save_best_features(top_features):
    #PATH = '/home/lucas/Documents/KYR/msc_thesis/vae-generator-for-particle-physics/analysis/features/'
    #FILE = f'features_top_{len(top_features)}.csv'
    
    PATH = '/home/lucas/Documents/KYR/msc_thesis/vae-generator-for-particle-physics/analysis/logging/'
    FILE = f'features_top_{len(top_features)}.log'
    
    helper_features = ['sig_mass', 'weight', 'row_number', 'file_number']
    all_features = top_features + helper_features

    top_features = list(top_features)
    with open(PATH + FILE, 'a') as file:
        file.write('[')
        for idx, top_feature in enumerate(top_features):
            if idx != len(top_features)-1:
                file.write(f'"{top_feature}", ')
            else: file.write(f'"{top_feature}"')
        file.write(']')
        file.write('\n')
    
    # with open(PATH + FILE, 'w', newline='') as file:
    #     writer = csv.writer(file)
    #     for feature in all_features:
    #         writer.writerow([feature])

    print(f"The list has been saved to {PATH+FILE}.")
    

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        # self.layers = nn.Sequential(
        #     nn.Linear(input_dim, 512),
        #     nn.BatchNorm1d(512),
        #     nn.ReLU(),
        #     nn.Linear(512, 1024),
        #     nn.BatchNorm1d(1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, 2048),
        #     nn.BatchNorm1d(2048),
        #     nn.ReLU(),
        #     nn.Linear(2048, 1024),
        #     nn.BatchNorm1d(1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, output_dim)
        # )
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            #nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            #nn.Dropout(0.1),
            nn.Linear(256, output_dim)
        )
        # self.layers = nn.Sequential(
        #     nn.Linear(input_dim, 64),
        #     nn.BatchNorm1d(64),
        #     nn.ReLU(),
        #     #nn.Dropout(0.1),
        #     nn.Linear(64, output_dim)
        # )
        # self.layers = nn.Sequential(
        #     nn.Linear(input_dim, 256),
        #     nn.BatchNorm1d(256),
        #     nn.ReLU(),
        #     nn.Dropout(0.3),  # Add dropout layer
        #     nn.Linear(256, 256),
        #     nn.BatchNorm1d(256),
        #     nn.ReLU(),
        #     nn.Dropout(0.3),  # Add dropout layer
        #     nn.Linear(256, output_dim)
        # )
        # self.layers = nn.Sequential(
        #     nn.Linear(input_dim, 512),
        #     nn.BatchNorm1d(512),
        #     nn.ReLU(),
        #     # nn.Dropout(0.1),
        #     nn.Linear(512, 256),
        #     nn.BatchNorm1d(256),
        #     nn.ReLU(),
        #     # nn.Dropout(0.1),
        #     nn.Linear(256, 128),
        #     nn.BatchNorm1d(128),
        #     nn.ReLU(),
        #     # nn.Dropout(0.1),
        #     nn.Linear(128, 64),
        #     nn.BatchNorm1d(64),
        #     nn.ReLU(),
        #     # nn.Dropout(0.1),
        #     nn.Linear(64, output_dim)
        # )
        # for layer in self.layers:
        #     if isinstance(layer, nn.Linear):
        #         init.xavier_uniform_(layer.weight)
        
    def forward(self, x):
        return self.layers(x)