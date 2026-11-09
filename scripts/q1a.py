"""Q1(a): Why does an MLP do poorly on CIFAR10?

Hypothesis: the MLP lacks spatial structure (locality and weight sharing).
Rival explanation: the MLP simply has too few parameters.

Fair comparison (fixed before running):
- same optimiser (Adam), batch size, number of epochs, normalisation, no augmentation
- each model tries the same 3 learning rates; the best one is picked on VALIDATION
- the epoch with the best validation accuracy is used for every model
- every model is trained with 3 seeds
- the test set is used only once per final model, at the very end

Predictions (written before running):
- MLP-1x about 50-55% validation accuracy, MLP-4x / MLP-16x about 55-58% (a plateau)
- CNN about 72-80%

Decision rule: if MLP-16x gains less than 3 points over MLP-1x, while the CNN is
more than 10 points above MLP-16x, then lack of spatial structure (not capacity)
is the main reason.
"""
import json

import matplotlib.pyplot as plt
import numpy as np

from common import ROOT, SEED, LOG_DIR, set_seed, load_data, fit, evaluate, test_eval, count_params
from models import make_cnn, make_mlp, find_hidden_size

EPOCHS = 40
BATCH_SIZE = 128
LEARNING_RATES = [3e-4, 1e-3, 3e-3]
SEEDS = [26007, 26008, 26009]
DROPOUT = 0.3
MODEL_NAMES = ["CNN", "MLP-1x", "MLP-4x", "MLP-16x", "MLP-1x-dropout"]

data = load_data()
cnn_params = count_params(make_cnn())


def build_model(name):
    """Return a fresh, untrained model for the given name."""
    if name == "CNN":
        return make_cnn()
    if name == "MLP-1x":
        return make_mlp(find_hidden_size(cnn_params))
    if name == "MLP-4x":
        return make_mlp(find_hidden_size(4 * cnn_params))
    if name == "MLP-16x":
        return make_mlp(find_hidden_size(16 * cnn_params))
    if name == "MLP-1x-dropout":
        return make_mlp(find_hidden_size(cnn_params), dropout=DROPOUT)


# ----------------------------------------------------------------------------
# Step 1: choose the learning rate for each model (validation only, one seed)
# ----------------------------------------------------------------------------
best_lr = {}
for name in MODEL_NAMES:
    best_val_acc = -1.0
    for lr in LEARNING_RATES:
        set_seed(SEED)                      # same starting point for every run
        model = build_model(name)
        history = fit(model, data, tag=f"q1a_tune_{name}_lr{lr}", epochs=EPOCHS,
                      lr=lr, batch_size=BATCH_SIZE, seed=SEED, verbose=False)
        print(f"{name}, lr={lr}: best val acc = {history['best_val_acc']:.4f}")
        if history["best_val_acc"] > best_val_acc:
            best_val_acc = history["best_val_acc"]
            best_lr[name] = lr
    print(f"--> {name}: chosen lr = {best_lr[name]}\n")


# ----------------------------------------------------------------------------
# Step 2: final runs with 3 seeds. All choices are fixed now, so we use the test set.
# ----------------------------------------------------------------------------
results = {}
first_seed_histories = {}      # kept for the plot

for name in MODEL_NAMES:
    train_accs, val_accs, test_accs = [], [], []

    for seed in SEEDS:
        set_seed(seed)
        model = build_model(name)
        history = fit(model, data, tag=f"q1a_final_{name}_seed{seed}", epochs=EPOCHS,
                      lr=best_lr[name], batch_size=BATCH_SIZE, seed=seed, verbose=False)

        # fit() puts the best-validation weights back into the model
        train_loss, train_acc = evaluate(model, data["x_train"], data["y_train"])
        test_loss, test_acc = test_eval(model, data, tag=f"q1a_{name}_seed{seed}",
                                        confirm_final=True)

        train_accs.append(train_acc)
        val_accs.append(history["best_val_acc"])
        test_accs.append(test_acc)
        if seed == SEEDS[0]:
            first_seed_histories[name] = history
        print(f"{name}, seed {seed}: train {train_acc:.4f}, val {history['best_val_acc']:.4f}, "
              f"test {test_acc:.4f}")

    results[name] = {
        "params": count_params(model),
        "lr": best_lr[name],
        "train_acc_mean": float(np.mean(train_accs)),
        "val_acc_mean": float(np.mean(val_accs)),
        "val_acc_std": float(np.std(val_accs)),
        "test_acc_mean": float(np.mean(test_accs)),
        "test_acc_std": float(np.std(test_accs)),
    }

with open(LOG_DIR / "q1a_summary.json", "w") as f:
    json.dump(results, f, indent=2)


# ----------------------------------------------------------------------------
# Step 3: results table
# ----------------------------------------------------------------------------
print("\nModel             Params      LR       Train    Val            Test")
for name in MODEL_NAMES:
    r = results[name]
    print(f"{name:<16}{r['params']:>9}  {r['lr']:<8} {r['train_acc_mean']:.4f}   "
          f"{r['val_acc_mean']:.4f}+-{r['val_acc_std']:.4f}  "
          f"{r['test_acc_mean']:.4f}+-{r['test_acc_std']:.4f}")


# ----------------------------------------------------------------------------
# Step 4: train and validation curves (first seed of each model)
# ----------------------------------------------------------------------------
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

fig, axes = plt.subplots(1, len(MODEL_NAMES), figsize=(4 * len(MODEL_NAMES), 3.5), sharey=True)
for i in range(len(MODEL_NAMES)):
    name = MODEL_NAMES[i]
    axes[i].plot(first_seed_histories[name]["train_acc"], label="train (running)")
    axes[i].plot(first_seed_histories[name]["val_acc"], label="validation")
    axes[i].set_title(name)
    axes[i].set_xlabel("epoch")
axes[0].set_ylabel("accuracy")
axes[0].legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "q1a_curves.png", dpi=150)


# ----------------------------------------------------------------------------
# Step 5: apply the decision rule (validation accuracy)
# ----------------------------------------------------------------------------
gain_from_capacity = results["MLP-16x"]["val_acc_mean"] - results["MLP-1x"]["val_acc_mean"]
gap_to_cnn = results["CNN"]["val_acc_mean"] - results["MLP-16x"]["val_acc_mean"]
print(f"\nGain from 16x more MLP parameters: {gain_from_capacity:.4f}")
print(f"Gap between CNN and MLP-16x:        {gap_to_cnn:.4f}")

if gain_from_capacity < 0.03 and gap_to_cnn > 0.10:
    print("Result: supports lack of spatial structure, not capacity.")
else:
    print("Result: does NOT clearly support the hypothesis. Discuss this honestly in the report.")