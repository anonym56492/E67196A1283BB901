# Adversarial Learning of Group and Individual Fair Representations (GIFair)
Supplementary material for submission Adversarial Learning of Group and Individual Fair Representations.

This folder contains the implementation of LAFTR and UNFAIR. Similar commands could be found as running GIFair.

## Installation
### Requirements
- Python >= 3.6
- [PyTorch](https://pytorch.org/) >= 1.4.0
- `pip install -r requirements.txt`

## Run LAFTR and UNFAIR
### Run with Specified Parameters
Run LAFTR training on Adult income dataset with group-fair coefficient 1.
```bash
python main.py --model LAFTR --fair-coeff 1 --dataset adult
```

Run UNFAIR training on Adult income dataset.
```bash
python main.py --model UNFAIR --dataset adult
```

Replace main.py the above commands with evaluate.py to run the evaluation with specific parameters.

### Run All the Experiments
To run all the experiments on LAFTR and UNFAIR, execute the script file linux_baseliens.sh (on a UNIX-based system) or win_baselines.bat (on a Windows system).