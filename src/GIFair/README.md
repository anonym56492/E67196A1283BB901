# Adversarial Learning of Group and Individual Fair Representations (GIFair)
Supplementary material for submission Adversarial Learning of Group and Individual Fair Representations.

## Installation
### Requirements
- Python >= 3.6
- [PyTorch](https://pytorch.org/) >= 1.4.0
- `pip install -r requirements.txt`

## Run GIFair
### Run with Specified Parameters
Run GIFair training on Adult income dataset with group-fair coefficient 1 and individual-fair coefficient 2.
```bash
python main.py --fair-coeff 1 --fair-coeff-individual 2 --dataset adult
```
Note that the coefficient for accuracy is fixed to 1. Other coefficients are default to 1. If a non-zero gamma is specified, the focal loss function will be enabled. If a non-zero lambda is specified, the balanced-yNN with specific lambda value will be used. More default options for each dataset is shown in `config.json`.

You can see more detailed options from
```bash
python main.py -h
```
Result files for training will be saved in `results/`. Saved models will be saved in `saved/`.

To evaluate the trained model with specific parameters, using the follwing command.
```bash
python evaluate.py --fair-coeff 1 --fair-coeff-individual 2 --dataset adult
```

### Run All the Experiments
To run all the experiments on GIFair, execute the script file linux_gifair.sh. Note that the random seed should be provided as a parameter of the script file. For example, to run linux_gifair.sh with random seed 0:
```bash
./linux_gifair.sh 0
```