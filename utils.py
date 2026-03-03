
import os
from PIL import Image
from torch.utils.data import Dataset, Subset, WeightedRandomSampler
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, classification_report
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import label_binarize
import torchvision.models as models
from torchvision.models import resnet18
import copy
import torch.nn as nn
import cv2
from torch.optim import LBFGS

def create_paths(model_name, augmented, env):
    suffix = "aug" if augmented else "no_aug"
    if env == "kaggle":
        # Pointing to Final_Models folder
        path_final_models = f"/kaggle/working/Final_Models/{model_name}_{suffix}"
        path_hackathon_checkpoints = f"/kaggle/working/hackathon_checkpoints/{model_name}_{suffix}"
        path_oct_models = f"/kaggle/working/oct_models/{model_name}_{suffix}"
        path_plots = f"/kaggle/working/plots/{model_name}_{suffix}"
        print("Running in Kaggle environment.")
    elif env == "colab":
        from google.colab import drive
        drive.mount('/content/drive')
        # Pointing to Final_Models folder
        path_final_models = f"/content/drive/MyDrive/Final_Models/{model_name}_{suffix}"
        path_hackathon_checkpoints = f"/content/drive/MyDrive/hackathon_checkpoints/{model_name}_{suffix}"
        path_plots = f"/content/drive/MyDrive/plots/{model_name}_{suffix}"
        path_oct_models = f"/content/drive/MyDrive/oct_models/{model_name}_{suffix}"
        print("Running in Colab environment.")
    elif env == "local":
        path_final_models = f"Final_Models/{model_name}_{suffix}"
        path_hackathon_checkpoints = f"hackathon_checkpoints/{model_name}_{suffix}"
        path_plots = f"plots/{model_name}_{suffix}"
        path_oct_models = f"oct_models/{model_name}_{suffix}"
        print("Running in local environment.")
    else:
        print("Invalid input. Defaulting to local.")
        path_final_models = f"Final_Models/{model_name}_{suffix}"
        path_hackathon_checkpoints = f"hackathon_checkpoints/{model_name}"
        path_plots = f"plots/{model_name}_{suffix}"
        path_oct_models = f"oct_models/{model_name}_{suffix}"
    print(f"The path are:\nFinal Models: {path_final_models}\nHackathon Checkpoints: {path_hackathon_checkpoints}\nPlots: {path_plots}\nOCT Models: {path_oct_models}")
    return path_final_models, path_hackathon_checkpoints, path_plots, path_oct_models

def create_dataframe(split, data_path):

    data = []
    split_path = os.path.join(data_path, split)

    for class_name in os.listdir(split_path):
        class_path = os.path.join(split_path, class_name)

        for img in os.listdir(class_path):
            img_path = os.path.join(class_path, img)
            data.append({
                "filepath": img_path,
                "label": class_name
            })

    return pd.DataFrame(data)


# Define the dataset
class OCTDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        if isinstance(idx, torch.Tensor):
            idx = idx.item()

        img_path = self.df.iloc[idx]["filepath"]
        label = self.df.iloc[idx]["label_encoded"]

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label)
    

# Create a class for early stopping
class Early_Stopping:
    def __init__(self, patience=3, min_delta=0.0005):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = None
        self.best_state = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, current_score, model):
        if self.best_score is None or current_score > self.best_score + self.min_delta:
            self.best_score = current_score
            self.best_state = copy.deepcopy(model.state_dict())
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                print("Early stopping triggered.")

class Early_Stopping_densenet:
    def __init__(self, patience=5, min_delta=0.0005, loss_threshold=1.5):
        self.patience = patience
        self.min_delta = min_delta
        self.loss_threshold = loss_threshold # If loss > 1.5x best_loss, we are overfitting
        self.best_score = None
        self.best_loss = float('inf')
        self.best_state = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, current_score, current_loss, model):
        if self.best_score is None:
            self.best_score = current_score
            self.best_loss = current_loss
            self.best_state = copy.deepcopy(model.state_dict())
            return

        # Boolean if f1 improved
        f1_improved = current_score > self.best_score + self.min_delta

        # Boolean if val loss increased a lot
        loss_increased = current_loss > (self.best_loss * self.loss_threshold)

        if f1_improved:
            if loss_increased:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
                    print("Early stopping triggered!")
            else:
                self.best_loss = current_loss
                self.best_score = current_score
                self.best_state = copy.deepcopy(model.state_dict())
                self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                print("Early stopping triggered!")


def get_subset(dataset, fraction=0.1):
    subset_size = int(len(dataset) * fraction)
    indices = torch.randperm(len(dataset))[:subset_size]
    return Subset(dataset, indices)

# Create a function for training for one epoch
def train_model(model, train_dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0

    all_preds, all_gold_labels = [], []

    for images, labels in tqdm(train_dataloader, desc="Training", unit="batch", position=0, leave=True):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad() # Reset optimizer grads

        outputs = model(images) # Do forward pass
        labels = labels.long()
        loss = criterion(outputs, labels)  # Compute Loss
        loss.backward()
        optimizer.step() # Update weights

        # Accumulate running loss
        running_loss += loss.item() * images.size(0)

        # Get predictions
        preds = outputs.argmax(dim=1)
        all_preds.append(preds.cpu()) # List of tensors
        all_gold_labels.append(labels.cpu()) # List of tensors

    # Make one tensor for labels
    all_preds = torch.cat(all_preds)
    all_gold_labels = torch.cat(all_gold_labels)

    # Compute F1-Score
    epoch_f1 = f1_score(y_pred=all_preds, y_true=all_gold_labels, average="macro")

    # Compute average loss
    epoch_loss = running_loss / len(train_dataloader.dataset)

    return epoch_loss, epoch_f1

def evaluate(model, val_dataloader, criterion, device):
    model.eval()
    running_loss = 0

    all_preds, all_gold_labels = [], []

    with torch.no_grad():
        for images, labels in tqdm(val_dataloader, desc="Evaluating", unit="batch", position=0, leave=True):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            labels = labels.long()
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            preds = outputs.argmax(dim=1)
            all_preds.append(preds.cpu())
            all_gold_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds)
    all_gold_labels = torch.cat(all_gold_labels)

    epoch_f1 = f1_score(y_pred=all_preds, y_true=all_gold_labels, average="macro")
    epoch_loss = running_loss / len(val_dataloader.dataset)

    return epoch_loss, epoch_f1

def loss_train_curve_plots(num_epochs, train_epoch_losses, train_epoch_f1s, val_epoch_losses, val_epoch_f1s,
                           stage, save_plot, model_name, augmented=False, path_plots=None):
    epochs = range(1, num_epochs + 1)
    plt.figure(figsize=(12, 4))

    epochs_xaxis = range(1, len(train_epoch_losses)+1)

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs_xaxis, train_epoch_losses, linestyle='-', marker='*', markersize=10, label='Train Loss')
    plt.plot(epochs_xaxis, val_epoch_losses, linestyle='-', marker='*', markersize=10, label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.xticks(range(1, len(train_epoch_losses)+1))
    plt.xlim(0.5, len(train_epoch_losses)+0.5)
    plt.title('Training Loss vs Validation Loss over epochs')
    plt.legend()
    plt.grid(True)

    # Plot F1-Score
    plt.subplot(1, 2, 2)
    plt.plot(epochs_xaxis, train_epoch_f1s, linestyle='-', marker='*', markersize=10, label='Train F1')
    plt.plot(epochs_xaxis, val_epoch_f1s, linestyle='-', marker='*', markersize=10, label='Val F1')
    plt.xlabel('Epochs')
    plt.ylabel('F1-Score')
    plt.xticks(range(1, len(train_epoch_losses)+1))
    plt.xlim(0.5, len(train_epoch_losses)+0.5)
    plt.title('Training F1-Score vs Validation F1-Score over epochs')
    plt.legend()
    plt.grid(True)

    # Save plot if triggered
    if save_plot:
        if stage is None:
            stage = 'full'

        os.makedirs(path_plots, exist_ok=True)

        # File name depends on augmentation
        if augmented:
            plot_path = f"{path_plots}/stage_{stage}_training_aug.png"
        else:
            plot_path = f"{path_plots}/stage_{stage}_training_not_aug.png"
        plt.savefig(plot_path)
        print(f"Plot saved to: {plot_path}")

    plt.show()
    plt.close()

def get_metrics(gold_labels, predicted_labels, results_dict):
    acc = accuracy_score(y_true=gold_labels, y_pred=predicted_labels)
    precision = precision_score(y_true=gold_labels, y_pred=predicted_labels, average='macro')
    recall = recall_score(y_true=gold_labels, y_pred=predicted_labels, average='macro')
    f1 = f1_score(y_true=gold_labels, y_pred=predicted_labels, average='macro')

    print(f"\n--- Metrics ---")
    print(f"Accuracy: {acc:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1 Score: {f1:.4f}")

    # Update the results dictionary
    results_dict.update({
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    })

    return acc, precision, recall, f1

def plot_cm_and_roc_single_stage(gold_labels, predicted_labels, probs, class_names, path_plot, augmented, model_name):
    num_classes = len(class_names)
    fig, axs = plt.subplots(1, 2, figsize=(14,6))

    # CONFUSION MATRIX
    cm = confusion_matrix(gold_labels, predicted_labels)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )

    disp.plot(cmap="Blues", ax=axs[0], colorbar=False)
    axs[0].set_title("Confusion Matrix")
    axs[0].tick_params(axis='x', rotation=45)

    # ROC CURVES
    binary_labels = label_binarize(
        gold_labels,
        classes=range(num_classes)
    )

    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(binary_labels[:, i], probs[:, i])
        roc_auc = auc(fpr, tpr)

        axs[1].plot(
            fpr,
            tpr,
            label=f"{class_names[i]} (AUC={roc_auc:.3f})"
        )

    axs[1].plot([0,1],[0,1], linestyle='--')
    axs[1].set_xlabel("False Positive Rate")
    axs[1].set_ylabel("True Positive Rate")
    axs[1].set_title("ROC Curves")
    axs[1].legend()
    axs[1].grid(True)

    fig.suptitle(f"Model: {model_name}", fontsize=14)
    plt.tight_layout()

    # Base path
    os.makedirs(path_plot, exist_ok=True)

    # File name depends on augmentation
    if augmented:
        plot_path = f"{path_plot}/{model_name}_cm_roc_aug.png"
    else:
        plot_path = f"{path_plot}/{model_name}_cm_roc_not_aug.png"

    # Save figure
    plt.savefig(plot_path, bbox_inches='tight')
    print(f"Plot saved to {plot_path}")

    plt.show()
    plt.close()

def save_best_model(model, model_name, stage_name, path_to_models):
    drive_base = path_to_models
    os.makedirs(drive_base, exist_ok=True)
    filename = f"{drive_base}/{model_name}_{stage_name}_best.pth"
    torch.save(model.state_dict(), filename)
    print("########################################################################################")
    print(f"Saved model to {filename}")
    print("########################################################################################")

def freeze_first_n_layers(model, stage=None):
    if isinstance(model, models.DenseNet):
        layer_order = [
            'features.denseblock1',
            'features.denseblock2',
            'features.denseblock3',
            'features.denseblock4',
            'classifier'
        ]
        model_type = "DenseNet"
    elif isinstance(model, models.ResNet):
        layer_order = [
            'layer1',
            'layer2',
            'layer3',
            'layer4',
            'fc'
        ]
        model_type = "ResNet"
    else:
        # Fallback for unknown models
        raise ValueError("Unknown model type. Please use either ResNet or DenseNet.")

    # First freeze everything
    for param in model.parameters():
        param.requires_grad = False

    if stage is None:
        # Unfreeze everything
        for param in model.parameters():
            param.requires_grad = True

        print("No layers are frozen, all layers are trainable.\n")
        return

    # Get index of selected stage
    stage_index = layer_order.index(stage)

    # Layers to UNFREEZE (from selected stage to end)
    layers_to_unfreeze = layer_order[stage_index:]

    # Apply unfreezing
    for name, param in model.named_parameters():
        if any(name.startswith(layer) for layer in layers_to_unfreeze):
            param.requires_grad = True

    # -------- DEBUG PRINTING --------
    print(f"Model type: {model_type}")
    print("Layer parameters that are NOT frozen at this stage are:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.requires_grad)

    print("")

    # Frozen layers are those before stage_index
    frozen_layers = layer_order[:stage_index]
    print(f"Frozen layers at this stage are: {frozen_layers}")

def save_checkpoint(model, optimizer, scheduler, stage, epoch,
                    train_losses, train_f1s, val_losses, val_f1s,
                    num_epochs, data_fraction, path):
    os.makedirs(path, exist_ok=True)
    checkpoint_file = os.path.join(path, f"{stage}_checkpoint_epoch_{epoch}.pt")
    torch.save({
        'stage': stage,
        'epoch': epoch,
        'num_epochs': num_epochs,
        'data_fraction': data_fraction,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_epoch_losses': train_losses,
        'train_epoch_f1s': train_f1s,
        'val_epoch_losses': val_losses,
        'val_epoch_f1s': val_f1s
    }, checkpoint_file)
    print(f"[DRIVE CHECKPOINT] Saved: {checkpoint_file}")

def load_latest_checkpoint(model, optimizer, scheduler, stage, path, device):
    if not os.path.exists(path):
        return model, optimizer, scheduler, 0, [], [], [], [], None, None

    checkpoints = [f for f in os.listdir(path) if f.startswith(f"{stage}_checkpoint_epoch_")]
    if not checkpoints:
        return model, optimizer, scheduler, 0, [], [], [], [], None, None

    latest_ckpt = sorted(checkpoints, key=lambda x: int(x.split("_")[-1].split(".")[0]))[-1]
    checkpoint = torch.load(os.path.join(path, latest_ckpt), map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch'] + 1

    print(f"[DRIVE CHECKPOINT] Loaded {latest_ckpt}, resuming from epoch {start_epoch}")

    # Return metrics + epoch info + fraction used
    return (model, optimizer, scheduler, start_epoch,
            checkpoint.get('train_epoch_losses', []),
            checkpoint.get('train_epoch_f1s', []),
            checkpoint.get('val_epoch_losses', []),
            checkpoint.get('val_epoch_f1s', []),
            checkpoint.get('num_epochs', None),
            checkpoint.get('data_fraction', None))

def test_model(model, test_dataloader, device, class_names, results_dict, model_name, augmented, path_plots):
    model.eval()

    all_preds, all_gold_labels, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in tqdm(test_dataloader, desc="Testing", unit="batch", position=0, leave=True):

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = probs.argmax(dim=1)

            all_preds.append(preds.cpu())
            all_gold_labels.append(labels.cpu())
            all_probs.append(probs.cpu())

    # Concatenate tensors
    all_preds = torch.cat(all_preds).numpy()
    all_gold_labels = torch.cat(all_gold_labels).numpy()
    all_probs = torch.cat(all_probs).numpy()

    # Metrics
    get_metrics(
        gold_labels=all_gold_labels,
        predicted_labels=all_preds,
        results_dict=results_dict
    )

    # Plot CM + ROC for THIS stage
    plot_cm_and_roc_single_stage(
        gold_labels=all_gold_labels,
        predicted_labels=all_preds,
        probs=all_probs,
        class_names=class_names,
        path_plot=path_plots,
        augmented=augmented,
        model_name=model_name
    )

    save_path = f"{path_plots}/{model_name}_results.pth"
    torch.save({
        "gold_labels": all_gold_labels,
        "predictions": all_preds,
        "probabilities": all_probs,
        "metrics": results_dict
    }, save_path)
    print(f"Raw predictions saved for comparison at: {save_path}")
    
    return all_gold_labels, all_preds, all_probs

def build_model(model_name):
    if 'resnet' in model_name:
        m = resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, 4)
    elif 'densenet' in model_name:
        m = models.densenet121(weights=None)
        m.classifier = nn.Linear(m.classifier.in_features, 4)
    return m

def extract_all_metrics(stages, checkpoint_path, model_name):
    # Initialize dictionary to hold all metrics
    metrics = {
        'train_loss': [],
        'val_loss': [],
        'train_f1': [],
        'val_f1': [],
        'stage_boundaries': [],
        'best_indices': []
    }

    cumulative_epochs = 0

    for stage in stages:
        # Get all checkpoint files for this stage
        files = [f for f in os.listdir(checkpoint_path) if str(stage) in f]
        
        # Guard clause in case a stage hasn't started/saved yet
        if not files:
            print(f"Skipping {stage}: No checkpoints found.")
            continue
            
        # Sort to find the last epoch
        files.sort(key=lambda x: int(x.split('epoch_')[-1].split('.pt')[0]))
        last_checkpoint_path = os.path.join(checkpoint_path, files[-1])
        
        # Load the data
        ckpt = torch.load(last_checkpoint_path, map_location='cpu')
        
        # Extract lists from checkpoint
        stage_val_losses = ckpt['val_epoch_losses']
        stage_val_f1s = ckpt['val_epoch_f1s']
        stage_train_losses = ckpt['train_epoch_losses']
        stage_train_f1s = ckpt['train_epoch_f1s']
        
        # Simulate EarlyStopping_densenet logic to find the best epoch index for this stage
        if "densenet" in model_name.lower():
            best_idx_in_stage = find_best_epoch_index_densenet(stage_val_f1s, stage_val_losses)
        elif "resnet" in model_name.lower():
            best_idx_in_stage = find_best_epoch_index_resnet(stage_val_f1s, stage_val_losses)
        else:
            raise ValueError("Wrong model name, please use either ResNet18 or DenseNet121.")

        metrics['best_indices'].append(cumulative_epochs + best_idx_in_stage)

        metrics['val_loss'].extend(stage_val_losses)
        metrics['val_f1'].extend(stage_val_f1s)
        metrics['train_loss'].extend(stage_train_losses)
        metrics['train_f1'].extend(stage_train_f1s)
        
        # Mark the end of this stage
        cumulative_epochs += len(stage_val_losses)
        metrics['stage_boundaries'].append(cumulative_epochs)

    return metrics

def find_best_epoch_index_densenet(f1s, losses, min_delta=0.0005, loss_threshold=1.5):
    best_score = -1
    best_loss = float('inf')
    best_idx = 0
    
    for i, (score, loss) in enumerate(zip(f1s, losses)):
        if i == 0:
            best_score, best_loss, best_idx = score, loss, i
            continue
            
        f1_improved = score > best_score + min_delta
        loss_increased = loss > (best_loss * loss_threshold)
        
        if f1_improved and not loss_increased:
            best_score = score
            best_loss = loss
            best_idx = i
            
    return best_idx

def find_best_epoch_index_resnet(f1s, losses=None, min_delta=0.0005):
    best_score = -float('inf')
    best_idx = 0
    
    for i, score in enumerate(f1s):
        # If current score is better than best + delta, it's the new best
        if score > best_score + min_delta:
            best_score = score
            best_idx = i
            
    return best_idx

def plot_multi_stage_metrics(metrics, stages, model_name, is_augmented):
    augmented = "Augmented" if is_augmented else "Base"
    
    # Extract data from dict for easier access
    v_loss = metrics['val_loss']
    t_loss = metrics['train_loss']
    v_f1 = metrics['val_f1']
    t_f1 = metrics['train_f1']
    boundaries = metrics['stage_boundaries']
    best_pts = metrics['best_indices']

    fig, ax1 = plt.subplots(figsize=(14, 8))

    # Plot Loss (Left Axis)
    # Training loss as a dashed, lighter line
    l1_t, = ax1.plot(t_loss, color='tab:red', linestyle='--', alpha=0.3, label='Train Loss')
    # Validation loss as a solid, bold line
    l1_v, = ax1.plot(v_loss, color='tab:red', linewidth=2, label='Val Loss')
    
    ax1.set_xlabel('Total Epochs (Across All Stages)', fontweight='bold')
    ax1.set_ylabel('Loss', color='tab:red', fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='tab:red')

    # Plot F1 (Secondary Axis)
    ax2 = ax1.twinx()
    # Training F1 as a dashed, lighter line
    l2_t, = ax2.plot(t_f1, color='tab:blue', linestyle='--', alpha=0.3, label='Train F1')
    # Validation F1 as a solid, bold line
    l2_v, = ax2.plot(v_f1, color='tab:blue', linewidth=2, label='Val F1')
    
    ax2.set_ylabel('F1 Score', color='tab:blue', fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    # Stage Boundaries and Labels
    for i, b in enumerate(boundaries):
        ax1.axvline(x=b-1, color='gray', linestyle=':', alpha=0.6)
        
        # Calculate midpoint
        prev_b = boundaries[i-1] if i > 0 else 0
        midpoint = prev_b + (b - prev_b) / 2
        
        stage_label = 'Fully_Unfrozen' if str(stages[i]) == 'None' else str(stages[i])
        
        ax1.text(midpoint, 1.01, stage_label, transform=ax1.get_xaxis_transform(),
                 ha='center', va='bottom', fontsize=9, fontweight='bold', 
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    # Highlight Best Models
    best_scat = ax2.scatter(best_pts, [v_f1[p] for p in best_pts], color='gold', 
                            marker='*', s=250, label='Selected Best Model', 
                            zorder=5, edgecolor='black')

    plt.title(f'Multi-Stage Training Progress | {model_name} | {augmented}', 
              pad=40, fontsize=16, fontweight="bold")
    
    # Combined Legend
    handles = [l1_t, l1_v, l2_t, l2_v, best_scat]
    labels = [h.get_label() for h in handles]
    ax1.legend(handles, labels, loc='upper left', ncol=2, fontsize='small')

    ax1.grid(True, alpha=0.3)
    fig.tight_layout()

def evaluate_val_split(model, loader, device, model_name, save_path):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc=f"Evaluating: {model_name}"):
            outputs = model(images.to(device))
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    target_names = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
    
    # Predictions
    prob_cols = [f"prob_{name}" for name in target_names]
    prob_df = pd.DataFrame(all_probs, columns=prob_cols)
    results_df = pd.DataFrame({
        'true_label': all_labels,
        'predicted_label': all_preds
    })
    full_stats_df = pd.concat([results_df, prob_df], axis=1)
    full_stats_df.to_csv(os.path.join(save_path, f"{model_name}_preds.csv"), index=False)

    # Classification report
    report_dict = classification_report(all_labels, all_preds, target_names=target_names, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    report_df.to_csv(os.path.join(save_path, f"{model_name}_report.csv"))

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    cm_df = pd.DataFrame(cm, index=target_names, columns=target_names)
    cm_df.to_csv(os.path.join(save_path, f"{model_name}_cm.csv"))

    # Return everything in a structured dictionary
    return {"full_stats": full_stats_df, "report": report_df,
            "confusion_matrix": cm_df, "class_names": target_names}

def generate_gradcam(model, image_tensor, target_class, device):
    model.eval()
    # Logic for DataParallel if needed
    base_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    
    activations = []
    gradients = []

    def forward_hook(module, input, output):
        activations.append(output)
    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    # Target the last dense block
    target_layer = base_model.features.denseblock4
    
    h_fwd = target_layer.register_forward_hook(forward_hook)
    h_bwd = target_layer.register_full_backward_hook(backward_hook)

    # Must allow gradients for the backward pass
    input_img = image_tensor.unsqueeze(0).to(device).requires_grad_(True)
    output = model(input_img)
    score = output[:, target_class]

    model.zero_grad()
    score.backward()

    grads = gradients[0].detach()
    acts = activations[0].detach()
    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * acts).sum(dim=1)).squeeze().cpu().numpy()
    
    # Cleanup hooks immediately
    h_fwd.remove()
    h_bwd.remove()
    
    return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

def visualize_gradcam(image_tensor, cam, true_class, pred_class, pred_conf, true_conf, save_path):
    mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
    img = image_tensor.permute(1, 2, 0).numpy()
    img = np.clip(img * std + mean, 0, 1)

    cam_resized = cv2.resize(cam, (img.shape[1], img.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title(f"T: {true_class} ({true_conf:.2f})\nP: {pred_class} ({pred_conf:.2f})")
    plt.subplot(1, 3, 2); plt.imshow(cam_resized, cmap="jet"); plt.title("Grad-CAM")
    plt.subplot(1, 3, 3); plt.imshow(0.5 * heatmap + 0.5 * img); plt.title("Overlay")
    
    if save_path: plt.savefig(save_path, bbox_inches='tight')
    plt.show()


def collect_hard_examples(final_model, loader, device, threshold=0.9):
    final_model.eval()
    hard_indices = []

    # Track class distribution of hard samples
    class_stats = {0: 0, 1: 0, 2: 0, 3: 0}

    with torch.no_grad():
        # Use enumerate to get the batch index
        for batch_idx, (images, labels) in enumerate(tqdm(loader, desc="Hard Mining")):
            images, labels = images.to(device), labels.to(device)

            outputs = final_model(images)
            probs = torch.softmax(outputs, dim=1)

            # Get max probability and predicted class
            confs, preds = torch.max(probs, dim=1)

            for i in range(len(labels)):
                # Calculate the ABSOLUTE index in the dataset
                global_idx = batch_idx * loader.batch_size + i

                # Logic: Prediction is wrong AND model was very sure
                if preds[i] != labels[i] and confs[i] > threshold:
                    hard_indices.append(global_idx)
                    class_stats[labels[i].item()] += 1

    print(f"\nFound {len(hard_indices)} 'Confidently Incorrect' samples.")
    print(f"Distribution of errors by True Class: {class_stats}")
    return hard_indices

def create_weighted_sampler(dataset, hard_indices):
    labels = dataset.df['label_encoded'].values
    weights = np.ones(len(labels), dtype=np.float64)

    weights[labels == 2] *= 1.4

    hard_indices_set = set(hard_indices)

    for idx in hard_indices:
        weights[idx] = max(weights[idx], 1.5)

    weights = np.nan_to_num(weights, nan=1.0, posinf=10.0)
    weights[weights <= 0] = 1e-7

    torch_weights = torch.from_numpy(weights).to(torch.double)

    return WeightedRandomSampler(weights=torch_weights, num_samples=len(torch_weights), replacement=True)

def unfreeze_last_block_densenet(final_model):
    # Freeze everything first
    for param in final_model.parameters():
        param.requires_grad = False

    # Unfreeze the final dense block and the final transition/norm layers
    for param in final_model.features.denseblock4.parameters():
        param.requires_grad = True
    for param in final_model.features.norm5.parameters():
        param.requires_grad = True

    # Unfreeze the classifier (DenseNet uses 'classifier', not 'fc')
    for param in final_model.classifier.parameters():
        param.requires_grad = True

    print("Unfroze classifier + denseblock4")

def fine_tune(final_model, train_loader, val_loader, optimizer, criterion, device, epochs, path):
    swa_path = os.path.join(path, "densenet_swa_refined.pth")

    swa_model = torch.optim.swa_utils.AveragedModel(final_model)
    # Start SWA after a few epochs of warming up the unmasked layers
    swa_start = 2
    print(epochs)
    for epoch in range(epochs):
        # Training
        final_model.train()
        total_train_loss = 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = final_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        # SWA Update
        if epoch >= swa_start:
            swa_model.update_parameters(final_model)
            print(f"--- SWA updated at end of Epoch {epoch+1} ---")

        # Validation
        final_model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = final_model(images)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()

        avg_val = total_val_loss / len(val_loader)
        print(f"Epoch {epoch+1}: Train Loss: {total_train_loss/len(train_loader):.4f} | Val Loss: {avg_val:.4f}")

    # SWA model
    torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)
    torch.save(swa_model.state_dict(), swa_path)
    return swa_model

def perform_temperature_scaling(logits, labels, device):
    # Convert numpy arrays from your inference to torch tensors
    logits = torch.from_numpy(logits).to(device)
    labels = torch.from_numpy(labels).long().to(device)

    # Initialize Temperature (T).
    # T > 1 "smoothes" the distribution (lowers confidence)
    # T < 1 "sharpens" it.
    temperature = nn.Parameter(torch.ones(1).to(device) * 1.1)

    # We only want to optimize 'temperature'
    optimizer = LBFGS([temperature], lr=0.01, max_iter=100)
    criterion = nn.CrossEntropyLoss()

    def eval_loop():
        optimizer.zero_grad()
        loss = criterion(logits / temperature, labels)
        loss.backward()
        return loss

    optimizer.step(eval_loop)

    return temperature.item()