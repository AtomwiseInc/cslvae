import torch
from torch.utils.data import Dataset
from typing import List

from cslvae.data import TorchMol, PackedTorchMol, NUM_NODE_FEATURES, NUM_EDGE_FEATURES


class SmilesDataset(Dataset):
    def __init__(self, smiles_list: List[str]):
        self.smiles_list = smiles_list

    @property
    def num_node_features(self) -> int:
        return NUM_NODE_FEATURES

    @property
    def num_edge_features(self) -> int:
        return NUM_EDGE_FEATURES

    def __len__(self) -> int:
        return len(self.smiles_list)

    def __getitem__(self, idx: int):
        smiles = self.smiles_list[idx]
        try:
            return {"idx": torch.tensor(idx, dtype=torch.long), "molecules": TorchMol(smiles)}
        except:
            return None

    @staticmethod
    def collate_fn(items):
        items = [item for item in items if item is not None]
        return {
            "idx": torch.stack([item["idx"] for item in items]),
            "molecules": PackedTorchMol([item["molecules"] for item in items]),
        }
