# Adversarial Learning of Group and Individual Fair Representations (GIFair)
Supplementary material for KDD 2023 submission Adversarial Learning of Group and Individual Fair Representations.

## Installation
### Requirements
- Linux with Python >= 3.6
- [PyTorch](https://pytorch.org/) >= 1.4.0
- `pip install -r requirements.txt`

## Run GIFair
### Run with Specified Parameters
Run DCFR model on Adult income dataset and conditional fairness task with fair coefficient 20.
```bash
python main.py --fair-coeff 1 --fair-coeff-individual 1 --dataset adult
```
Note that the coefficient for accuracy is fixed to 1. Other coefficients are default to 1. If a non-zero gamma is specified, the focal loss function will be enabled. More default options for each dataset is shown in config.json.

You can see more options from
```bash
python main.py -h
```
Result files will be saved in `results/`. Saved models will be saved in `saved/`.
