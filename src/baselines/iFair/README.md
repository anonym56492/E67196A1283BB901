# Adversarial Learning of Group and Individual Fair Representations (GIFair)
Supplementary material for submission Adversarial Learning of Group and Individual Fair Representations.

This folder contains the implementation of iFair.

## Installation
### Requirements
- Python >= 3.6
- [PyTorch](https://pytorch.org/) >= 1.4.0
- `pip install -r requirements.txt`

## Run iFair
### Run with Specified Parameters
Run iFair training on Adult income dataset with individual-fair coefficient 1.
```bash
python main.py --fair-coeff-individual 1 --dataset adult
```

Replace main.py the above commands with evaluate.py to run the evaluation with specific parameters.

### Run All the Experiments
To run all the experiments on iFair, execute the script file linux_baseliens.sh (on a UNIX-based system) or win_baselines.bat (on a Windows system).