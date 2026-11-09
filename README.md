# ML-assignment-3
The third and final assignment for the Machine Learning Fall course for the year 2026.


# ML Assignment 3: CIFAR10

Name: Renzen Lama | Roll no: M26CSA007 | Seed: 26007

## Environment
```
pip install -r requirements.txt
```

## Reproduction (updated as questions are completed)
```
python make_split.py        # creates/verifies splits/cifar10_split_seed26007.npz
# q1.py, q2.py, q3.py, q4.py added as each question is finished
```

## Protocol
- One stratified 45,000/5,000 train/validation split, seed = 26007, saved in `splits/`.
- Normalisation statistics computed from the 45k training images only.
- All selection uses validation only. The test set is accessed solely through
  `common.test_eval(..., confirm_final=True)`, which logs to `logs/final_test_results.jsonl`.
- Per-run training curves are in `logs/<tag>.json`.

## Reused components
Tracked here with the question where each was first introduced (fill in as we go).