# CSLVAE

This is the PyTorch implementation of the **C**ombinatorial **S**ynthesis **L**ibrary **V**ariational **A**uto-**E**ncoder (CSLVAE) described in the [NeurIPS 2022 paper](https://arxiv.org/abs/2211.04468).

### Installation

```buildoutcfg
# Set up conda environment
export ENVNAME=cslvae
conda create -n $ENVNAME -c rdkit -c conda-forge python=3.8.10 pip rdkit=2020.09.1
conda activate $ENVNAME

# Install the corresponding versions of torch scatter and torch sparse
pip install torch==1.12.0+cu102
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.12.0+cu102.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-1.12.0+cu102.html

# Git clone repo and pip install
git clone https://github.com/AtomwiseInc/cslvae.git
cd cslvae
pip install . --user

# Set up to use conda environment as the notebook kernel
conda install -c anaconda ipykernel
python -m ipykernel install --user --name=$ENVNAME
```

### Fine-tune on a user-provided combinatorial synthesis library (CSL)

```buildoutcfg
# Fine-tune on a new CSL using the pre-trained weights
train_cslvae.py \
--reaction_smarts_path <reaction_smarts_path> \
--synthon_smiles_path <synthon_smiles_path> \
--config configs/train_cslvae_config.yaml \
--weights_path models/cslvae.pt \
--output_dir <output_dir>
```

### Example notebook

Please see the [notebook](https://github.com/AtomwiseInc/cslvae/tree/main/notebooks/cslvae_demo.ipynb) for usage examples of the auto-encoding exercises considered in the paper.

### Required data format for CSLs

CSLVAE requires a reaction SMARTS file and a synthon SMILES file to construct the CSL, represented as a PyTorch dataset `CSLDataset`. These files are expected to be tab-delimited and the columns must be formatted like shown in the provided examples. The naming convention for the `reaction_id` and `synthon_id` fields shown below is not a requirement; users are free to assign string/int IDs for these fields to their liking. A subset of the 2022 Enamine REAL library in the required format can be requested from [Enamine](https://enamine.net/contact), who have graciously agreed to provide such access. These files can be utilized to reproduce the results in the notebook.

##### Reaction SMARTS file format

| reaction_id | num_rgroups | smarts                               |
| :----       | :----       | :----                                |
| rxn_0       | 2           | [U]-[\*:1].[U]-[\*:2]>>[\*:1]-[\*:2] |
| ...         | ...         | ...                                  |

##### Synthon SMILES file format

| synthon_id | reaction_id | rgroup | smiles                |
| :----      | :----       | :----  | :----                 |
| syn_0      | rxn_0       | 0      | CN1CCC(N[U])C11CCCNC1 |
| syn_1      | rxn_0       | 0      | [U]Nc1ccc2NCCCc2c1    |
| ...        | ...         | ...    | ...                   |
| syn_79     | rxn_0       | 1      | [U]C(=O)C1CC11CC1     |
| ...        | ...         | ...    | ...                   |
