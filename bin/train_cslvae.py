#!/usr/bin/env python

import argparse
import os
import random
import shutil
import string
import torch
import yaml
from rdkit import RDLogger

from cslvae.dataset import CSLDataset
from cslvae.nn import CSLVAE


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run CSLVAE training.", formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--reaction_smarts_path", type=str, action="store", required=True,
        help="Path to reaction SMARTS file",
    )
    parser.add_argument(
        "--synthon_smiles_path", type=str, action="store", required=True,
        help="Path to synthon SMILES file",
    )
    parser.add_argument(
        "--config", type=str, action="store", required=True,
        help="Path to YAML file specifying the configuration.",
    )
    parser.add_argument(
        "--output_dir", type=str, action="store", required=False, default=os.getcwd(),
        help="Path to output directory. If not provided, defaults to current working directory.",
    )
    parser.add_argument(
        "--run_id", type=str, action="store", required=False, default=None,
        help="Run ID, used in constructing output directories and in logs to tensorboard. If not "
             "provided, a random 16-character alpha-numeric run_id will be generated.",
    )
    parser.add_argument(
        "--weights_path", type=str, action="store", required=False, default=None,
        help="Path to existing model weights. If no provided, weights are randomly initialized.",
    )
    parser.add_argument(
        "--device", type=str, action="store", required=False, default="",
        help="Device used for model training."
    )
    parser.add_argument(
        "--disable_rdkit_logs", type=bool, action="store", required=False, default=True,
        help="Setting this to False will include RDKit logs in the logfile.",
    )
    return parser.parse_args()


def parse_config(args):
    with open(args.config, "r") as fp:
        config = yaml.load(fp, Loader=yaml.CLoader)
    return config


def main():
    # Parse input arguments and config
    args = parse_arguments()
    config = parse_config(args)

    # Use the user-provided run id or generate a random 16-character alpha-numeric id
    if args.run_id is None:
        run_id = "".join(random.choice(string.ascii_lowercase + string.digits) for _ in range(16))
    else:
        run_id = str(args.run_id)

    # Create output directories, start logs, and save a copy of the config file (in case needed as
    # reference)
    outdir = os.path.join(args.output_dir, run_id)
    os.makedirs(os.path.join(outdir, "checkpoints"))
    shutil.copy(args.config, os.path.join(outdir, "config.yaml"))

    if bool(args.disable_rdkit_logs):
        RDLogger.DisableLog("rdApp.*")

    if len(args.device) > 0:
        device = str(args.device)
    else:
        if torch.cuda.device_count() > 0:
            device = "cuda:0"
        else:
            device = "cpu"

    print(f"Run ID: {run_id}.")
    print(f"All outputs will be written to: {outdir}.")
    print(f"GPU count: {torch.cuda.device_count()}. CPU count: {os.cpu_count()}.")
    print(f"Training on device: {device}.")

    # Get config parts
    config_model = config.get("model", dict())
    config_train = config.get("training", dict())

    # Load dataset
    dataset = CSLDataset(args.reaction_smarts_path, args.synthon_smiles_path)

    print(f"Loaded combinatorial synthesis library.")
    print(f"Number of reactions: {dataset.num_reactions:,}.")
    print(f"Number of synthons: {dataset.num_synthons:,}.")
    print(f"Number of products: {len(dataset):,}.")

    # Load model
    cslvae = CSLVAE(**config_model).to(device)
    if args.weights_path is not None:
        print(f"Loading model from {args.weights_path}.")
        checkpoint_state_dict = torch.load(args.weights_path, map_location=device)
        cslvae.load_state_dict(checkpoint_state_dict["model_state_dict"])

    print("Architecture:")
    print(cslvae)
    print(f"Parameter count: {sum(p.numel() for p in cslvae.parameters()):,}.")

    # Fit
    cslvae.fit(dataset, config_train, outdir)


if __name__ == "__main__":
    main()
