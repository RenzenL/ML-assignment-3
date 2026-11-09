"""Create (once) and verify the stratified 45k/5k CIFAR10 split.

Run:  python make_split.py
Commit the resulting splits/*.npz file to the repository.
"""
import hashlib

import numpy as np

from common import ROLL_NO, SEED, SPLIT_PATH, CLASSES, load_cifar10_raw, make_split

if __name__ == "__main__":
    path = make_split()
    s = np.load(path)
    tr, va = s["train_idx"], s["val_idx"]
    _, y, _, _ = load_cifar10_raw()

    assert len(tr) == 45000 and len(va) == 5000
    assert len(set(tr) & set(va)) == 0, "train/val overlap"
    assert len(set(tr) | set(va)) == 50000

    print(f"Roll no: {ROLL_NO} -> seed {SEED}")
    print(f"Split file: {path}")
    print(f"sha256: {hashlib.sha256(open(path, 'rb').read()).hexdigest()}")
    print(f"{'class':<12}{'train':>8}{'val':>8}")
    for c, name in enumerate(CLASSES):
        print(f"{name:<12}{(y[tr] == c).sum():>8}{(y[va] == c).sum():>8}")