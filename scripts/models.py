"""Model definitions. First introduced in Q1, reused in later questions."""
import torch.nn as nn


def make_cnn():
    """Small CNN: 3 conv blocks, then a small classifier. About 357k parameters."""
    return nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),    # 32x32 -> 16x16
        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 16x16 -> 8x8
        nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 8x8 -> 4x4
        nn.Flatten(),
        nn.Linear(128 * 4 * 4, 128), nn.ReLU(),
        nn.Linear(128, 10),
    )


def make_mlp(hidden, dropout=0.0):
    """MLP with two hidden layers of the same width. dropout=0.0 means no dropout."""
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(3 * 32 * 32, hidden), nn.ReLU(), nn.Dropout(dropout),
        nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
        nn.Linear(hidden, 10),
    )


def mlp_param_count(hidden):
    """Parameters of make_mlp(hidden):
    first layer 3072*h + h, second layer h*h + h, last layer 10*h + 10."""
    return hidden * hidden + 3084 * hidden + 10


def find_hidden_size(target_params):
    """Find the hidden width whose MLP has the number of parameters closest to the target."""
    best_hidden = 1
    best_difference = abs(mlp_param_count(1) - target_params)
    for hidden in range(2, 5000):
        difference = abs(mlp_param_count(hidden) - target_params)
        if difference < best_difference:
            best_difference = difference
            best_hidden = hidden
    return best_hidden