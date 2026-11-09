"""Q1(b): Does an MLP learn useful structure before it starts memorising?

Setting: a large MLP (enough capacity to memorise) is trained on a 10,000 image
subset of the training set where 40% of the labels are replaced by WRONG random labels.
Wrong labels cannot be generalised, so fitting them is pure memorisation.

Each epoch we measure four things:
1. clean_train : accuracy on training images whose label is correct
2. noisy_fit   : accuracy on the noisy images against their WRONG assigned labels (memorisation)
3. noisy_true  : accuracy on the same noisy images against their TRUE labels (structure)
4. val         : accuracy on the validation set (true labels, never trained on)

Prediction: val and noisy_true rise first while noisy_fit stays near chance (10%).
Later noisy_fit climbs (memorisation), and val stops improving or drops.

No test set is used here.
"""
import json

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from common import ROOT, SEED, LOG_DIR, set_seed, load_data, make_loader, evaluate, count_params
from models import make_mlp

SUBSET_SIZE = 10000
NOISE_FRACTION = 0.4
HIDDEN = 1024
EPOCHS = 100
LR = 1e-3
BATCH_SIZE = 128
MEMORISATION_THRESHOLD = 0.30     # chance level on wrong labels is about 0.10~

set_seed(SEED)
data = load_data()

# ---- build the training subset (random, not stratified) ----
random_order = torch.randperm(len(data["x_train"]))[:SUBSET_SIZE]
x = data["x_train"][random_order]
true_y = data["y_train"][random_order]

# ---- replace the labels of the first 40% with wrong labels ----
n_noisy = int(SUBSET_SIZE * NOISE_FRACTION)
train_y = true_y.clone()
shift = torch.randint(1, 10, (n_noisy,))             # 1..9, so the new label is never the true one
train_y[:n_noisy] = (true_y[:n_noisy] + shift) % 10

x_noisy, x_clean = x[:n_noisy], x[n_noisy:]
true_y_noisy, true_y_clean = true_y[:n_noisy], true_y[n_noisy:]
wrong_y_noisy = train_y[:n_noisy]

# ---- model and optimiser (no dropout, no weight decay: we WANT it to memorise) ----
model = make_mlp(HIDDEN)
print("MLP hidden width:", HIDDEN, "parameters:", count_params(model))
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loader = make_loader(x, train_y, BATCH_SIZE, shuffle=True)

history = {"clean_train": [], "noisy_fit": [], "noisy_true": [], "val": []}

for epoch in range(1, EPOCHS + 1):
    # ---- one training epoch ----
    model.train()
    for batch_x, batch_y in loader:
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(model(batch_x), batch_y)
        loss.backward()
        optimizer.step()

    # ---- measurements (evaluate() switches the model to eval mode) ----
    clean_train = evaluate(model, x_clean, true_y_clean)[1]
    noisy_fit = evaluate(model, x_noisy, wrong_y_noisy)[1]
    noisy_true = evaluate(model, x_noisy, true_y_noisy)[1]
    val = evaluate(model, data["x_val"], data["y_val"])[1]

    history["clean_train"].append(clean_train)
    history["noisy_fit"].append(noisy_fit)
    history["noisy_true"].append(noisy_true)
    history["val"].append(val)

    if epoch % 5 == 0 or epoch == 1:
        print(f"epoch {epoch:3d} | clean_train {clean_train:.3f} | noisy_fit {noisy_fit:.3f} | "
              f"noisy_true {noisy_true:.3f} | val {val:.3f}")

# ----------------------------------------------------------------------------
# Decide: when did memorisation start, and when was validation accuracy best?
# ----------------------------------------------------------------------------
best_val_epoch = history["val"].index(max(history["val"])) + 1

memorisation_epoch = None
for i in range(EPOCHS):
    if history["noisy_fit"][i] > MEMORISATION_THRESHOLD:
        memorisation_epoch = i + 1
        break

print("\nBest validation epoch:", best_val_epoch, "val acc:", round(max(history["val"]), 3))
print("Memorisation starts (noisy_fit >", MEMORISATION_THRESHOLD, ") at epoch:", memorisation_epoch)
print("At the best validation epoch: noisy_fit =", round(history["noisy_fit"][best_val_epoch - 1], 3),
      ", noisy_true =", round(history["noisy_true"][best_val_epoch - 1], 3))
print("At the last epoch: noisy_fit =", round(history["noisy_fit"][-1], 3),
      ", noisy_true =", round(history["noisy_true"][-1], 3), ", val =", round(history["val"][-1], 3))

if memorisation_epoch is None:
    print("Result: the model never memorised the wrong labels. Train longer or use a bigger model.")
elif best_val_epoch < memorisation_epoch:
    print("Result: validation accuracy peaked BEFORE memorisation began, so useful structure was learned first.")
else:
    print("Result: validation was still improving when memorisation began. Discuss this honestly.")

# ---- save numbers and plot ----
with open(LOG_DIR / "q1b_memorisation.json", "w") as f:
    json.dump(history, f, indent=2)

FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

epochs = list(range(1, EPOCHS + 1))
plt.figure(figsize=(7, 4.5))
plt.plot(epochs, history["clean_train"], label="clean train images")
plt.plot(epochs, history["noisy_fit"], label="noisy images vs WRONG labels (memorisation)")
plt.plot(epochs, history["noisy_true"], label="noisy images vs TRUE labels")
plt.plot(epochs, history["val"], label="validation")
plt.axvline(best_val_epoch, color="gray", linestyle="--")
if memorisation_epoch is not None:
    plt.axvline(memorisation_epoch, color="red", linestyle=":")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(FIG_DIR / "q1b_memorisation.png", dpi=150)