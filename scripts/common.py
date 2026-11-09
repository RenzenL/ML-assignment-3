"""
Shared setup for ML Assignment 3 (CIFAR10).

This file contains all the reusable boilerplate for Q1-Q4. We handle:
- Seeding for reproducibility
- A saved, stratified 45k/5k train/validation split
- Data loading and safe normalization
- A standard training loop with metrics logging
- A guarded test-set evaluator to prevent accidental data leakage
"""
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

SEED = 26007

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
SPLIT_PATH = ROOT / "splits" / f"cifar10_split_seed{SEED}.npz"
LOG_DIR = ROOT / "logs"
CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


# ----------------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------------
def set_seed(seed: int = SEED, deterministic: bool = True) -> None:
    """Lock down all random number generators for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Automatically pick the best hardware accelerator available."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps") # For Apple Silicon Macs
    return torch.device("cpu")


# ----------------------------------------------------------------------------
# Data Processing and Splitting
# ----------------------------------------------------------------------------
def load_cifar10_raw():
    """Download and load CIFAR10 into memory. Returns raw numpy arrays."""
    from torchvision.datasets import CIFAR10
    
    train_dataset = CIFAR10(root=str(DATA_DIR), train=True, download=True)
    test_dataset = CIFAR10(root=str(DATA_DIR), train=False, download=True)
    
    return (
        train_dataset.data, np.array(train_dataset.targets, dtype=np.int64),
        test_dataset.data, np.array(test_dataset.targets, dtype=np.int64)
    )


def make_split(force: bool = False) -> Path:
    """
    Create a stratified 45k/5k train/validation split. 
    We save this to disk so we always use the exact same split across runs.
    """
    from sklearn.model_selection import train_test_split
    
    # Skip creation if we already have the file (unless forced to recreate)
    if SPLIT_PATH.exists() and not force:
        return SPLIT_PATH
        
    _, y_train_full, _, _ = load_cifar10_raw()
    all_indices = np.arange(len(y_train_full))
    
    train_idx, val_idx = train_test_split(
        all_indices, 
        test_size=5000, 
        random_state=SEED, 
        stratify=y_train_full
    )
    
    SPLIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(SPLIT_PATH, train_idx=train_idx, val_idx=val_idx, seed=SEED)
    return SPLIT_PATH


def _to_tensor(x_uint8: np.ndarray, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Convert raw images to normalized PyTorch tensors (N, C, H, W)."""
    # Convert to float, scale to [0, 1], and rearrange channels
    x = torch.from_numpy(x_uint8).float().div_(255.0).permute(0, 3, 1, 2)
    return (x - mean) / std


def load_data(normalize: bool = True) -> dict:
    """
    Load the data using our saved split. 
    Crucially, normalisation stats are computed ONLY on the 45k training set
    so we don't leak information from the validation or test sets.
    """
    make_split()
    x_full, y_full, x_test_raw, y_test_raw = load_cifar10_raw()
    
    saved_split = np.load(SPLIT_PATH)
    train_idx = saved_split["train_idx"]
    val_idx = saved_split["val_idx"]

    # Extract the training subset to compute our normalization stats
    x_train_raw = x_full[train_idx]
    
    if normalize:
        # Calculate mean and std deviation across N, H, W (leaving the 3 color channels)
        channel_means = torch.from_numpy(x_train_raw).float().div(255.0).mean(dim=(0, 1, 2))
        channel_stds = torch.from_numpy(x_train_raw).float().div(255.0).std(dim=(0, 1, 2))
        
        # Reshape to (1, 3, 1, 1) so it broadcasts correctly against (N, C, H, W) images
        mean = channel_means.view(1, 3, 1, 1)
        std = channel_stds.view(1, 3, 1, 1)
    else:
        mean, std = torch.zeros(1, 3, 1, 1), torch.ones(1, 3, 1, 1)

    return {
        "x_train": _to_tensor(x_train_raw, mean, std),
        "y_train": torch.from_numpy(y_full[train_idx]),
        "x_val": _to_tensor(x_full[val_idx], mean, std),
        "y_val": torch.from_numpy(y_full[val_idx]),
        "x_test": _to_tensor(x_test_raw, mean, std),
        "y_test": torch.from_numpy(y_test_raw),
        "mean": mean, 
        "std": std,
    }


def make_loader(x, y, batch_size=128, shuffle=False, seed=SEED) -> DataLoader:
    """Helper to quickly spin up a DataLoader from tensors."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    
    return DataLoader(
        TensorDataset(x, y), 
        batch_size=batch_size,
        shuffle=shuffle, 
        generator=generator if shuffle else None
    )


# ----------------------------------------------------------------------------
# Model Utilities
# ----------------------------------------------------------------------------
def count_params(model: nn.Module, trainable_only: bool = True) -> int:
    """Count the number of parameters in a model."""
    return sum(
        p.numel() for p in model.parameters() 
        if p.requires_grad or not trainable_only
    )


@torch.no_grad()
def predict_logits(model, x, device=None, batch_size=1000) -> torch.Tensor:
    """Run data through the model in chunks to avoid blowing up memory."""
    device = device or get_device()
    model.eval().to(device)
    
    predictions = []
    for i in range(0, len(x), batch_size):
        batch = x[i : i + batch_size].to(device)
        predictions.append(model(batch).cpu())
        
    return torch.cat(predictions)


@torch.no_grad()
def evaluate(model, x, y, device=None, batch_size=1000):
    """Calculate standard loss and accuracy metrics for a given dataset."""
    logits = predict_logits(model, x, device, batch_size)
    loss = nn.functional.cross_entropy(logits, y).item()
    accuracy = (logits.argmax(1) == y).float().mean().item()
    
    return loss, accuracy


# ----------------------------------------------------------------------------
# Training Loop
# ----------------------------------------------------------------------------
def fit(model, data, tag, epochs=30, lr=1e-3, weight_decay=0.0, batch_size=128,
        optimizer="adam", momentum=0.9, seed=SEED, x_train=None, y_train=None,
        keep_best=True, device=None, verbose=True):
    """
    Standard training loop that handles backprop, evaluation, and logging.
    
    If 'keep_best' is True, we monitor the validation accuracy and restore 
    the best weights before returning the model.
    """
    device = device or get_device()
    set_seed(seed)
    model.to(device)
    
    # Allow overriding training data (useful for experiments like noisy labels)
    train_x = data["x_train"] if x_train is None else x_train
    train_y = data["y_train"] if y_train is None else y_train
    
    loader = make_loader(train_x, train_y, batch_size, shuffle=True, seed=seed)

    # Setup the optimizer
    if optimizer == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer == "sgd":
        opt = torch.optim.SGD(
            model.parameters(), lr=lr, momentum=momentum,
            weight_decay=weight_decay, nesterov=True
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer}")

    # Set up our history dictionary to log metrics
    history = {
        "tag": tag, "params": count_params(model), "epochs": epochs, "lr": lr,
        "weight_decay": weight_decay, "batch_size": batch_size,
        "optimizer": optimizer, "seed": seed,
        "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []
    }
    
    best_val_acc = -1.0
    best_state_dict = None
    best_epoch = -1

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, total_correct, num_samples = 0.0, 0, 0
        
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            opt.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = nn.functional.cross_entropy(logits, batch_y)
            
            loss.backward()
            opt.step()
            
            total_loss += loss.item() * len(batch_y)
            total_correct += (logits.argmax(1) == batch_y).sum().item()
            num_samples += len(batch_y)
            
        # Evaluate on the validation set at the end of the epoch
        val_loss, val_acc = evaluate(model, data["x_val"], data["y_val"], device)
        
        train_loss = total_loss / num_samples
        train_acc = total_correct / num_samples
        
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        
        # Keep track of our best model weights
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            
        if verbose:
            print(
                f"[{tag}] Epoch {epoch:3d}/{epochs} | "
                f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}"
            )

    # Save metrics and restore best weights
    history["best_val_acc"] = best_val_acc
    history["best_epoch"] = best_epoch
    
    if keep_best and best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_DIR / f"{tag}.json", "w") as f:
        json.dump(history, f, indent=2)
        
    return history


# ----------------------------------------------------------------------------
# Test Set (Guarded Evaluation)
# ----------------------------------------------------------------------------
def test_eval(model, data, tag, confirm_final=False, device=None):
    """
    Evaluate on the official test set. 
    
    This function explicitly requires `confirm_final=True` to run. This acts as 
    a safety guardrail so you don't accidentally overfit to the test set while 
    tuning your models!
    """
    if not confirm_final:
        raise RuntimeError(
            "Test set is for final evaluation only! "
            "Pass confirm_final=True only once all hyperparameters are fixed."
        )
        
    loss, acc = evaluate(model, data["x_test"], data["y_test"], device)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    # Append results so every final run is traceable
    with open(LOG_DIR / "final_test_results.jsonl", "a") as f:
        result_entry = {
            "tag": tag, 
            "test_loss": loss, 
            "test_acc": acc,
            "params": count_params(model)
        }
        f.write(json.dumps(result_entry) + "\n")
        
    return loss, acc