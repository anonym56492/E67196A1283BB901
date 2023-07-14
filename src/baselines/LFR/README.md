# Adversarial Learning of Group and Individual Fair Representations (GIFair)
Supplementary material for submission Adversarial Learning of Group and Individual Fair Representations.

This folder contains the implementation of LFR.

## Installation
### Requirements
- Python >= 3.6
- scipy
- numba
- scikit-learn
- pandas

## Run LFR
### Run with Specified Parameters
Run LFR on Adult income dataset with group-fair coefficient 1, individual-fair coefficient 2 and random seed 0.
```bash
python lfr.py adult 1 2 0
```

### Run All the Experiments
To run all the experiments on LFR, execute the script file run.sh (on a UNIX-based system) or run.bat (on a Windows system).