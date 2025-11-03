from __future__ import annotations
import os, random, json, math, itertools
from pathlib import Path
import time
import warnings # To suppress specific warnings if needed

import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import StratifiedKFold, ParameterGrid
from sklearn.utils import class_weight
from sklearn.metrics import (
    precision_recall_fscore_support, confusion_matrix, balanced_accuracy_score,
    roc_auc_score
)

import matplotlib.pyplot as plt

# Attempt to import SMOTE
try:
    from imblearn.over_sampling import SMOTE
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [WARN] imbalanced-learn library not found. SMOTE functionality will be disabled.")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [WARN] Please install it if you intend to use SMOTE: pip install imbalanced-learn")


# ------------------------------------------------------------------
#  0.  Reproducibility helpers
# ------------------------------------------------------------------
def set_seed(seed:int=42) -> None:
    """Sets random seeds for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

# ------------------------------------------------------------------
#  1.  IO
# ------------------------------------------------------------------
DATA_DIR   = Path('data_files')
SAVE_DIR   = Path('./results_grid_1') # Updated save directory name
SAVE_DIR.mkdir(exist_ok=True)

BEST_CONFIG_FILE = SAVE_DIR / 'best_config.json'

print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [INFO] Loading data...")
try:
    X_baseline = np.load(DATA_DIR/'baseline_data_New.npy')
    X_screw    = np.load(DATA_DIR/'screws_data_New.npy')
    X_crack    = np.load(DATA_DIR/'cracks_data_New.npy')
except FileNotFoundError as e:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [ERROR] Data file not found: {e}. Exiting.")
    exit()
except Exception as e:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [ERROR] Failed to load data: {e}")
    exit()

y_baseline = np.zeros (len(X_baseline), dtype=np.int64)
y_screw    = np.ones  (len(X_screw)   , dtype=np.int64)
y_crack    = np.full  (len(X_crack)   , 2 , dtype=np.int64)
X_original = np.concatenate([X_baseline, X_screw, X_crack], axis=0)
y_original = np.concatenate([y_baseline, y_screw, y_crack], axis=0)

if X_original.shape[0] == 0:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [ERROR] No data loaded. Exiting.")
    exit()
print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [INFO] Original dataset shape: {X_original.shape}")
print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [INFO] Original class distribution: {dict(zip(*np.unique(y_original, return_counts=True)))}")

# ------------------------------------------------------------------
#  2.  Augmentation utilities
# ------------------------------------------------------------------
def gaussian_noise(x, std):
    return x + np.random.normal(0, std, x.shape).astype(np.float32)
def jitter(x, max_shift:int):
    shift = np.random.randint(-max_shift, max_shift + 1) if max_shift > 0 else 0
    return np.roll(x, shift, axis=1)
def scaling(x, rng):
    scale_shape = (x.shape[0], 1) if x.ndim == 2 else (x.shape[0], 1, 1) if x.ndim == 3 else (1,)
    return x * np.random.uniform(*rng, scale_shape).astype(np.float32)

def augment_sample(x_sample: np.ndarray, std: float, jitt: int, rng: tuple):
    if x_sample.ndim != 2:
        raise ValueError(f"augment_sample expects 2D input (Sensors, Sequence), got {x_sample.ndim}D")
    x_aug = gaussian_noise(x_sample, std)
    x_aug = jitter(x_aug, jitt)
    x_aug = scaling(x_aug, rng)
    return x_aug

def expand_class(X: np.ndarray, k: int, std: float, jitt: int, rng: tuple):
    if X.shape[0] == 0 or k <= 0:
        return np.empty((0, *X.shape[1:]), dtype=np.float32)
    augmented_list = []
    for i in range(X.shape[0]):
        original_sample = X[i]
        for _ in range(k):
            augmented_sample = augment_sample(original_sample, std, jitt, rng)
            augmented_list.append(augmented_sample)
    if not augmented_list: return np.empty((0, *X.shape[1:]), dtype=np.float32)
    return np.stack(augmented_list, axis=0)

# ------------------------------------------------------------------
#  3.  Dataset + dataloader
# ------------------------------------------------------------------
class FRFDataset(Dataset):
    def __init__(self, X, y, mean: float = 0.0, std: float = 1.0):
        if std < 1e-9: std = 1.0
        self.X = ((X - mean) / std).astype(np.float32)
        self.y = y.astype(np.int64)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])

# ------------------------------------------------------------------
#  4.  Model Definition
# ------------------------------------------------------------------
class SensorEncoder(nn.Module):
    def __init__(self, embed=128):
        super().__init__()
        chs=(1, 32, 64, 128, embed)
        mods=[]
        for i, (c_in, c_out) in enumerate(zip(chs[:-1], chs[1:])):
            mods += [nn.Conv1d(c_in, c_out, kernel_size=3, padding=1),
                     nn.BatchNorm1d(c_out), nn.ReLU()]
            if i < len(chs) - 2:
                mods.append(nn.MaxPool1d(kernel_size=2))
        mods.append(nn.AdaptiveAvgPool1d(output_size=1))
        self.encoder = nn.Sequential(*mods)
    def forward(self, x):
        out = self.encoder(x)
        return out.squeeze(-1)

class TransformerSHM(nn.Module):
    def __init__(self, n_sensor=28, L=150, embed=128, n_classes=3, n_heads=4, n_layers=2, dropout=0.1, ff_dim=512):
        super().__init__()
        self.n_sensor = n_sensor
        self.L = L
        self.embed = embed
        self.sensor_enc = SensorEncoder(embed)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed, nhead=n_heads, dim_feedforward=ff_dim,
            dropout=dropout, batch_first=True, activation='relu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.q_fc = nn.Sequential(nn.Linear(embed, embed), nn.ReLU())
        self.mha = nn.MultiheadAttention(
            embed_dim=embed, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.classifier = nn.Sequential(
            nn.Linear(embed, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, n_classes)
        )

    def forward(self, x):
        B = x.size(0)
        x_flat = x.view(B * self.n_sensor, 1, self.L)
        sensor_features = self.sensor_enc(x_flat).view(B, self.n_sensor, self.embed)
        h = self.transformer(sensor_features)
        q = self.q_fc(h.mean(dim=1)).unsqueeze(1)
        out, attn = self.mha(q, h, h, need_weights=True)
        logits = self.classifier(out.squeeze(1))
        return logits, attn.squeeze(1)

# ------------------------------------------------------------------
#  5.  Train utilities
# ------------------------------------------------------------------
def train_epoch(model, loader, optimizer, criterion, l1_lambda, device, scaler):
    model.train()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    for batch in loader:
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
            logits, attn = model(inputs)
            loss = criterion(logits, labels) + l1_lambda * attn.abs().sum()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * len(labels)
        total_correct += (logits.argmax(1) == labels).sum().item()
        total_samples += len(labels)
    return total_loss / total_samples, total_correct / total_samples

def validate(model, loader, criterion, device):
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    y_true, y_pred, y_prob, attn_list = [], [], [], []
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
            for batch in loader:
                inputs, labels = batch
                inputs, labels = inputs.to(device), labels.to(device)
                logits, attn = model(inputs)
                loss = criterion(logits, labels)
                total_loss += loss.item() * len(labels)
                total_correct += (logits.argmax(1) == labels).sum().item()
                total_samples += len(labels)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(logits.argmax(1).cpu().numpy())
                y_prob.extend(torch.softmax(logits, dim=1).cpu().numpy())
                attn_list.append(attn.mean(dim=0).cpu().numpy())
    mean_attn = np.mean(attn_list, axis=0) if attn_list else None
    return (
        total_loss / total_samples, total_correct / total_samples,
        np.array(y_true), np.array(y_pred), np.array(y_prob), mean_attn
    )

def compute_metrics(y_true, y_pred, y_prob, n_classes=3):
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=range(n_classes), zero_division=0)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    conf_mat = confusion_matrix(y_true, y_pred, labels=range(n_classes))
    auc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro') if len(np.unique(y_true)) > 1 else 0.0
    return {
        'precision': precision.tolist(),
        'recall': recall.tolist(),
        'f1': f1.tolist(),
        'balanced_acc': balanced_acc,
        'conf_matrix': conf_mat.tolist(),
        'roc_auc': auc
    }

# ------------------------------------------------------------------
#  6.  Grid Search
# ------------------------------------------------------------------
param_grid = {
    'n_heads': [2, 4],
    'n_layers': [1, 2],
    'ff_dim': [256, 512],
    'lr': [1e-4, 3e-4],
    'l1_lambda': [1e-5, 1e-4]
}
# Fixed
embed_size = 128
dropout_rate = 0.1
batch_size = 32
grid_epochs = 50  # Shorter for grid
grid_patience = 15
full_epochs = 100
full_patience = 30
grid_folds = 3
full_reps = 3
full_folds = 10

grid_configs = list(ParameterGrid(param_grid))
print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [INFO] Grid search with {len(grid_configs)} configurations.")

best_score = -1
best_config = None
grid_results = {}

for config_idx, config in enumerate(grid_configs, 1):
    config_dir = SAVE_DIR / f'grid_config_{config_idx}'
    config_dir.mkdir(exist_ok=True)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [GRID] Config {config_idx}/{len(grid_configs)}: {config}")
    
    per_model_metrics = []
    per_model_attns = []
    
    set_seed(42)
    skf = StratifiedKFold(n_splits=grid_folds, shuffle=True, random_state=42)
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_original, y_original), 1):
        X_train, y_train = X_original[train_idx], y_original[train_idx]
        X_val, y_val = X_original[val_idx], y_original[val_idx]
        
        if IMBLEARN_AVAILABLE:
            smote = SMOTE(random_state=42)
            X_train_flat = X_train.reshape(len(X_train), -1)
            X_train_smote, y_train_smote = smote.fit_resample(X_train_flat, y_train)
            X_train = X_train_smote.reshape(-1, 28, 150)
            y_train = y_train_smote
        else:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [WARN] SMOTE not applied.")
        
        class_counts = np.bincount(y_train, minlength=3)
        aug_factors = [20 if count > 0 else 0 for count in class_counts]
        aug_params = [(119.86, 3, (0.9, 1.1)), (119.37, 3, (0.9, 1.1)), (150.0, 5, (0.85, 1.15))]
        
        X_aug = [X_train]
        y_aug = [y_train]
        for cls, factor, params in zip(range(3), aug_factors, aug_params):
            X_cls = X_train[y_train == cls]
            if len(X_cls) > 0 and factor > 0:
                X_aug_cls = expand_class(X_cls, factor, *params)
                X_aug.append(X_aug_cls)
                y_aug.append(np.full(len(X_aug_cls), cls, dtype=np.int64))
        
        X_train_aug = np.concatenate(X_aug, axis=0)
        y_train_aug = np.concatenate(y_aug, axis=0)
        
        mean, std = X_train_aug.mean(), X_train_aug.std() or 1.0
        
        train_ds = FRFDataset(X_train_aug, y_train_aug, mean, std)
        val_ds = FRFDataset(X_val, y_val, mean, std)
        
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = TransformerSHM(n_sensor=28, L=150, embed=embed_size, n_classes=3,
                               n_heads=config['n_heads'], n_layers=config['n_layers'],
                               dropout=dropout_rate, ff_dim=config['ff_dim']).to(device)
        
        class_weights = torch.tensor(class_weight.compute_class_weight('balanced', classes=np.unique(y_train_aug), y=y_train_aug), dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights if class_weights.sum() > 0 else None)
        
        optimizer = optim.Adam(model.parameters(), lr=config['lr'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=grid_patience//2)
        scaler = torch.cuda.amp.GradScaler(enabled=device.type == 'cuda')
        
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        
        train_losses, train_accs, val_losses, val_accs = [], [], [], []
        
        for epoch in range(1, grid_epochs + 1):
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, config['l1_lambda'], device, scaler)
            val_loss, val_acc, y_true, y_pred, y_prob, mean_attn = validate(model, val_loader, criterion, device)
            
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            
            scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= grid_patience:
                    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [GRID] Config {config_idx} Fold {fold_idx} Early stopping at epoch {epoch}")
                    break
            
            if epoch % 10 == 0 or epoch == 1:
                print(f"Config {config_idx} Fold {fold_idx} Ep {epoch:03d}/{grid_epochs} Tr Ls: {train_loss:.4f} Ac: {train_acc:.4f} | Vl Ls: {val_loss:.4f} Ac: {val_acc:.4f}")
        
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            torch.save(model.state_dict(), config_dir / f'best_model_fold_{fold_idx}.pth')
        
        _, _, y_true, y_pred, y_prob, mean_attn = validate(model, val_loader, criterion, device)
        fold_metrics = compute_metrics(y_true, y_pred, y_prob)
        per_model_metrics.append(fold_metrics)
        if mean_attn is not None:
            per_model_attns.append(mean_attn)
        
        # Save fold results
        with open(config_dir / f'fold_{fold_idx}_metrics.json', 'w') as f:
            json.dump(fold_metrics, f, indent=4)
        
        # Per-fold learning curve plot
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(train_accs, label='Train Acc')
        plt.plot(val_accs, label='Val Acc')
        plt.legend()
        plt.savefig(config_dir / f'fold_{fold_idx}_curves.png')
        plt.close()
    
    # Aggregate grid config results
    avg_bal_acc = np.mean([m['balanced_acc'] for m in per_model_metrics])
    grid_results[config_idx] = {'config': config, 'avg_bal_acc': avg_bal_acc, 'metrics': per_model_metrics}
    
    with open(config_dir / 'aggregate_metrics.json', 'w') as f:
        json.dump(grid_results[config_idx], f, indent=4)
    
    if avg_bal_acc > best_score:
        best_score = avg_bal_acc
        best_config = config
        with open(BEST_CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [GRID] New best config {config_idx}: BalAcc {avg_bal_acc:.4f}")

print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [GRID] Best config: {best_config} with score {best_score:.4f}")

# ------------------------------------------------------------------
#  7.  Full CV with Best Config
# ------------------------------------------------------------------
if best_config is None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [ERROR] No best config found. Exiting.")
    exit()

full_dir = SAVE_DIR / 'full_cv_best'
full_dir.mkdir(exist_ok=True)

print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [FULL] Running full CV with best config: {best_config}")

per_model_metrics_list = []
per_model_scores_list = []
per_model_curves_list = []
overall_conf_matrix = np.zeros((3, 3), dtype=int)
sensor_mean = None
sensor_std_across_models = None
total_models_trained = 0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_seeds = [42, 123, 789]

for rep in range(1, full_reps + 1):
    rep_dir = full_dir / f'rep_{rep}'
    rep_dir.mkdir(exist_ok=True)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ========= Repetition {rep}/{full_reps} =========")
    skf_seed = model_seeds[rep - 1]
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [Rep {rep}] SKF seed: {skf_seed}, Model training seed for this rep: {skf_seed}")
    set_seed(skf_seed)
    
    skf = StratifiedKFold(n_splits=full_folds, shuffle=True, random_state=skf_seed)
    
    for fold in range(1, full_folds + 1):
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] --- Rep {rep}, Fold {fold}/{full_folds} ---")
        train_idx, val_idx = list(skf.split(X_original, y_original))[fold - 1]
        X_train, y_train = X_original[train_idx], y_original[train_idx]
        X_val, y_val = X_original[val_idx], y_original[val_idx]
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [Rep {rep}, F {fold}] Orig Train: {X_train.shape}, Orig Val: {X_val.shape}")
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [Rep {rep}, F {fold}] Orig Train dist: {dict(zip(*np.unique(y_train, return_counts=True)))}")
        
        if IMBLEARN_AVAILABLE:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [Rep {rep}, F {fold}] Applying SMOTE with default k_neighbors...")
            X_train_flat = X_train.reshape(len(X_train), -1)
            smote = SMOTE(random_state=skf_seed)
            X_train_smote, y_train_smote = smote.fit_resample(X_train_flat, y_train)
            X_train = X_train_smote.reshape(-1, 28, 150)
            y_train = y_train_smote
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [Rep {rep}, F {fold}] After SMOTE Train: {X_train.shape}")
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [Rep {rep}, F {fold}] After SMOTE Train dist: {dict(zip(*np.unique(y_train, return_counts=True)))}")
        
        class_counts = np.bincount(y_train, minlength=3)
        aug_factors = [20, 20, 20]
        aug_params = [(119.86, 3, (0.9, 1.1)), (119.37, 3, (0.9, 1.1)), (150.0, 5, (0.85, 1.15))]
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [Rep {rep}, F {fold}] Custom Aug factors (on post-SMOTE data): {aug_factors}")
        for cls, factor, count in zip(range(3), aug_factors, class_counts):
            print(f"  - Custom Aug Class {cls} (post-SMOTE): Factor={factor}, Samples={count}")
        
        X_aug = [X_train]
        y_aug = [y_train]
        for cls, factor, params in zip(range(3), aug_factors, aug_params):
            X_cls = X_train[y_train == cls]
            if len(X_cls) > 0 and factor > 0:
                X_aug_cls = expand_class(X_cls, factor, *params)
                X_aug.append(X_aug_cls)
                y_aug.append(np.full(len(X_aug_cls), cls, dtype=np.int64))
        
        X_train_aug = np.concatenate(X_aug, axis=0)
        y_train_aug = np.concatenate(y_aug, axis=0)
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [Rep {rep}, F {fold}] Combined Train (SMOTE + Custom Aug) shape: {X_train_aug.shape}")
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [Rep {rep}, F {fold}] Combined Train class dist: {dict(zip(*np.unique(y_train_aug, return_counts=True)))}")
        
        mean, std = X_train_aug.mean(), X_train_aug.std() or 1.0
        
        train_ds = FRFDataset(X_train_aug, y_train_aug, mean, std)
        val_ds = FRFDataset(X_val, y_val, mean, std)
        
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        
        class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train_aug), y=y_train_aug)
        if len(class_weights) == 3:
            class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [Rep {rep}, F {fold}] Applied Class weights: {class_weights.cpu().numpy()}")
        else:
            class_weights = None
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [Rep {rep}, F {fold}] Final train data is well-balanced. Consider not using class weights or using uniform weights.")
        
        model = TransformerSHM(n_sensor=28, L=150, embed=embed_size, n_classes=3,
                               n_heads=best_config['n_heads'], n_layers=best_config['n_layers'],
                               dropout=dropout_rate, ff_dim=best_config['ff_dim']).to(device)
        
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(model.parameters(), lr=best_config['lr'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=full_patience//2)
        scaler = torch.cuda.amp.GradScaler(enabled=device.type == 'cuda')
        
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        
        train_losses, train_accs, val_losses, val_accs = [], [], [], []
        
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] --- Training: Rep {rep}, Fold {fold}, Seed {skf_seed} ---")
        for epoch in range(1, full_epochs + 1):
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, best_config['l1_lambda'], device, scaler)
            val_loss, val_acc, y_true, y_pred, y_prob, mean_attn = validate(model, val_loader, criterion, device)
            
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            
            scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= full_patience:
                    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Early stopping at ep {epoch}")
                    break
            
            if epoch % 10 == 0 or epoch == 1:
                print(f"Rep {rep}, Fold {fold}, Seed {skf_seed}, Ep [{epoch:03d}/{full_epochs}] Tr Ls: {train_loss:.4f} Ac: {train_acc:.4f} | Vl Ls: {val_loss:.4f} Ac: {val_acc:.4f}")
        
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            torch.save(best_model_state, rep_dir / f'best_model_fold_{fold}.pth')
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Rep {rep}, Fold {fold}, Seed {skf_seed}: Loaded best model (Val Acc: {max(val_accs):.4f})")
        
        _, val_acc, y_true, y_pred, y_prob, mean_attn = validate(model, val_loader, criterion, device)
        fold_metrics = compute_metrics(y_true, y_pred, y_prob)
        fold_metrics['accuracy'] = val_acc
        per_model_metrics_list.append(fold_metrics)
        if mean_attn is not None:
            per_model_scores_list.append(mean_attn)
        per_model_curves_list.append({
            'train_losses': np.array(train_losses),
            'train_accs': np.array(train_accs),
            'val_losses': np.array(val_losses),
            'val_accs': np.array(val_accs)
        })
        total_models_trained += 1
        
        overall_conf_matrix += np.array(fold_metrics['conf_matrix'])
        
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [Rep {rep}, F {fold}] Val Met: Acc={val_acc:.4f}, BalAcc={fold_metrics['balanced_acc']:.4f}, AUC={fold_metrics['roc_auc']:.4f}")
    
# Aggregate full results
if per_model_metrics_list:
    overall_metrics = {
        'accuracy': np.mean([m['accuracy'] for m in per_model_metrics_list if 'accuracy' in m]),
        'balanced_acc': np.mean([m['balanced_acc'] for m in per_model_metrics_list]),
        'precision': np.mean([m['precision'] for m in per_model_metrics_list], axis=0).tolist(),
        'recall': np.mean([m['recall'] for m in per_model_metrics_list], axis=0).tolist(),
        'f1': np.mean([m['f1'] for m in per_model_metrics_list], axis=0).tolist(),
        'roc_auc': np.mean([m['roc_auc'] for m in per_model_metrics_list]),
        'conf_matrix': overall_conf_matrix.tolist()
    }
    with open(full_dir / 'overall_metrics.json', 'w') as f:
        json.dump(overall_metrics, f, indent=4)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [INFO] Aggregating results and generating plots...")
    print("=== Overall Metrics (Aggregated across All Folds from All Repetitions) ===")
    print(json.dumps(overall_metrics, indent=2))

# Sensor importance
valid_per_model_scores = [s for s in per_model_scores_list if s is not None]
if valid_per_model_scores:
    sensor_mean = np.mean(valid_per_model_scores, axis=0)
    sensor_std_across_models = np.std(valid_per_model_scores, axis=0)

# Plots (same as original)
class_names = ['Baseline', 'Screw', 'Crack']
plt.rcParams.update({'font.size': 11, 'figure.autolayout': True, 'figure.dpi': 100})

if sensor_mean is not None and sensor_std_across_models is not None:
    plt.figure(figsize=(12, 5)); num_sensors = len(sensor_mean); sensor_indices = np.arange(1, num_sensors + 1)
    plt.bar(sensor_indices, sensor_mean, yerr=sensor_std_across_models, capsize=4, color='skyblue', edgecolor='black', linewidth=0.7, error_kw={'alpha':0.6,'lw':1,'ecolor':'gray'})
    plt.title(f'Overall Sensor Importance ({total_models_trained} Models)'); plt.xlabel('Sensor Index'); plt.ylabel('Mean Attention Weight (Error bars: Std across models)')
    plt.xticks(sensor_indices[::2]); plt.grid(axis='y', linestyle='--', alpha=0.6); plt.xlim(0.5, num_sensors + 0.5)
    plt.savefig(full_dir / 'overall_sensor_ranking.eps'); plt.close()
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [PLOT] Saved overall_sensor_ranking.eps")

valid_model_metrics = [m for m in per_model_metrics_list if m is not None]
model_accuracies = [m['accuracy'] for m in valid_model_metrics if 'accuracy' in m]
if model_accuracies:
    plt.figure(figsize=(6, 5))
    plt.boxplot(model_accuracies, patch_artist=True, showmeans=True, boxprops={'facecolor':'lightgreen','alpha':0.8,'edgecolor':'black'}, medianprops={'color':'red','linewidth':1.5}, meanprops={'marker':'^','markersize':8,'markeredgecolor':'black','markerfacecolor':'blue'})
    plt.ylabel('Model Accuracy'); plt.xlabel(f'Models (N={len(model_accuracies)})'); plt.xticks([1],[' '])
    plt.title(f'Accuracy Distribution Across {len(model_accuracies)} Models'); plt.grid(axis='y',linestyle='--',alpha=0.4)
    plt.ylim(bottom=max(0,min(model_accuracies)-0.05 if model_accuracies else 0), top=min(1.05,max(model_accuracies)+0.05 if model_accuracies else 1.05))
    plt.savefig(full_dir / 'model_accuracy_boxplot.eps'); plt.close()
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [PLOT] Saved model_accuracy_boxplot.eps")

conf_matrix_np = np.array(overall_metrics['conf_matrix']) if 'conf_matrix' in overall_metrics else None
if conf_matrix_np is not None and conf_matrix_np.ndim==2 and conf_matrix_np.shape==(3,3):
    plt.figure(figsize=(6,5)); im=plt.imshow(conf_matrix_np,interpolation='nearest',cmap=plt.cm.Blues)
    plt.colorbar(im,label='Number of Samples'); tick_marks=np.arange(len(class_names))
    plt.xticks(tick_marks,class_names,rotation=45,ha="right"); plt.yticks(tick_marks,class_names)
    thresh = conf_matrix_np.max()/2.
    for i,j in itertools.product(range(conf_matrix_np.shape[0]),range(conf_matrix_np.shape[1])):
        plt.text(j,i,format(int(conf_matrix_np[i,j]),'d'),ha="center",va="center",color="white" if conf_matrix_np[i,j]>thresh else "black",fontsize=10,weight='bold')
    plt.ylabel('True Label'); plt.xlabel('Predicted Label'); plt.title(f'Overall Confusion Matrix ({total_models_trained} Models Aggregated)')
    plt.savefig(full_dir /'confusion_matrix.eps'); plt.close()
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [PLOT] Saved confusion_matrix.eps")

if len(valid_per_model_scores) > 1 :
    mean_of_model_scores = np.nanmean(np.stack(valid_per_model_scores), axis=0)
    std_of_model_scores  = np.nanstd(np.stack(valid_per_model_scores), axis=0)
    plt.figure(figsize=(12,5)); num_sensors = len(mean_of_model_scores); sensor_indices = np.arange(1,num_sensors+1)
    for score_array_fold in valid_per_model_scores:
        if score_array_fold is not None and score_array_fold.ndim==1 and len(score_array_fold)==num_sensors:
            plt.plot(sensor_indices,score_array_fold,alpha=0.25,lw=0.8,color='lightblue',label='_nolegend_')
    plt.plot(sensor_indices,mean_of_model_scores,color='black',linestyle='-',linewidth=2.0,label='Mean Importance (across models)')
    plt.fill_between(sensor_indices, mean_of_model_scores - std_of_model_scores, mean_of_model_scores + std_of_model_scores, color='gray',alpha=0.3,label='Mean ± 1 Std Dev (across models)')
    plt.xlabel('Sensor Index'); plt.ylabel('Mean Attention Weight per Model'); plt.xticks(sensor_indices[::2])
    plt.title(f'Sensor Importance Variability Across {len(valid_per_model_scores)} Models'); plt.grid(axis='y',linestyle='--',alpha=0.4)
    plt.legend(loc='best',fontsize='small'); plt.xlim(0.5,num_sensors+0.5)
    plt.savefig(full_dir /'sensor_importance_variability.eps'); plt.close()
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [PLOT] Saved sensor_importance_variability.eps")

valid_model_curves = [c for c in per_model_curves_list if c is not None and len(c['train_losses'])>0]
if valid_model_curves:
    min_len = min(len(c['train_losses']) for c in valid_model_curves if len(c['train_losses'])>0)
    if min_len > 0:
        overall_curves = {key: np.mean([c[key][:min_len] for c in valid_model_curves],axis=0) for key in valid_model_curves[0].keys()}
        epochs_axis = np.arange(1,min_len+1)
        plt.figure(figsize=(10,8)); plt.subplot(2,1,1)
        plt.plot(epochs_axis,overall_curves['train_losses'],label='Avg Train Loss',color='blue',linewidth=1.5)
        plt.plot(epochs_axis,overall_curves['val_losses'],label='Avg Val Loss',color='orange',linewidth=1.5)
        plt.ylabel('Cross-Entropy Loss'); plt.title(f'Average Model Learning Curves ({len(valid_model_curves)} Models Averaged)')
        plt.grid(linestyle='--',alpha=0.6); plt.legend(); plt.ylim(bottom=0); plt.subplot(2,1,2)
        plt.plot(epochs_axis,overall_curves['train_accs'],label='Avg Train Accuracy',color='blue',linewidth=1.5)
        plt.plot(epochs_axis,overall_curves['val_accs'],label='Avg Val Accuracy',color='orange',linewidth=1.5)
        plt.ylabel('Accuracy'); plt.xlabel('Epoch'); plt.grid(linestyle='--',alpha=0.6); plt.legend(); plt.ylim(0,1.05)
        plt.savefig(full_dir /'overall_model_learning_curves.eps'); plt.close()
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [PLOT] Saved overall_model_learning_curves.eps")
    else: print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [WARN] Min epoch length 0 for curves.")
else: print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [WARN] No valid model curves for overall plot.")

print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] [DONE] Script finished.")
