import argparse
import json
import os

import numpy as np
import torch

import pdb

from gifair.datasets.adult import AdultDataset
from gifair.datasets.compas import CompasDataset
from gifair.datasets.german import GermanDataset
from gifair.models.laftr import LAFTR
from gifair.utils.runner import Runner


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="random seed")

    parser.add_argument(
        "--dataset",
        type=str,
        default="adult",
        choices=["adult", "compas", "german"],
        help="Adult income / COMPAS / German dataset",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="LAFTR", ## LAFTR -> GIFair
        choices=["LAFTR"],
        help="LAFTR model",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="DP",
        choices=["CF", "DP", "EO"],
        help="fairness task",
    )
    parser.add_argument("--lr", type=float, default=None, help="learning rate")
    parser.add_argument("--k", type=int, default=10, help="k value")
    parser.add_argument("--zdim", type=int, default=None, help="zdim value")
    parser.add_argument("--gamma", type=float, default=0, help="gamma value")
    parser.add_argument(
        "--fair-coeff", type=float, default=None, help="fair coefficient"
    )
    parser.add_argument(
        "--fair-coeff-individual", type=float, default=None, help="fair coefficient individual"
    )
    parser.add_argument("--batch-size", type=int, default=None, help="batch size")
    parser.add_argument("--epoch", type=int, default=None, help="epoch number")
    parser.add_argument(
        "--optim",
        type=str,
        default=None,
        choices=["Adadelta", "Adam", "RMSprop", "Adagrad"],
        help="optimization algorithm",
    )
    parser.add_argument(
        "--aud-steps",
        type=int,
        default=None,
        help="audition step for adversarial learning",
    )
    parser.add_argument(
        "--aud-individual-steps",
        type=int,
        default=None,
        help="audition step for adversarial learning- individual",
    )
    parser.add_argument("--lambda", type=float, default=0, help="gamma_k value")
    parser.add_argument(
        "--tensorboard", action="store_true", help="whether to plot in tensorboard"
    )

    args = parser.parse_args()
    with open(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json"), "r"
    ) as f:
        config = json.loads(f.read())
        f.close()
    config = config[args.dataset]
    args = vars(args)
    for arg in ["lr", "k", "fair_coeff", "fair_coeff_individual", "batch_size", "epoch", "optim", "aud_steps", "aud_individual_steps", "zdim"]:
        if args[arg] is None:
            args[arg] = config[arg]
    config.update(args)
    print(config)
    return config


def main():
    config = parse_args()
    torch.random.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    if config["dataset"] == "adult":
        dataset = AdultDataset()
    elif config["dataset"] == "compas":
        dataset = CompasDataset()
    elif config["dataset"] == "german":
        dataset = GermanDataset()

    if config["model"] == "LAFTR":
        model = LAFTR(config)
    else:
        print(f"Model {config['model']} not supported")
        exit()

    runner = Runner(dataset, model, config, evaluate=True)

    # pdb.set_trace()
    test_res = runner.test(400)
    print(f"tested: DP = {test_res['test']['DP']}, yNN = {test_res['test']['YNN-k-10-l-0']}, acc = {test_res['test']['acc']}")


    # runner.finetune("last")
    # runner.test("finetune_last_best")


if __name__ == "__main__":
    main()
