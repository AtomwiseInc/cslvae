import torch
from rdkit import Chem
from rdkit.Chem.rdchem import Atom, Bond, Mol
from torch import Tensor
from typing import List, Tuple

from .atom_bond_features import get_atom_features, get_bond_features


class TorchMol:
    def __init__(self, smiles: str, remove_hydrogens: bool = True):
        super().__init__()
        self.smiles = str(smiles)
        fn = lambda mol: Chem.RemoveHs(mol) if remove_hydrogens else mol
        self.mol = fn(Chem.MolFromSmiles(self.smiles))
        self.node_features, self.edge_features, self.edge_index, self.graph_index = self.process()

    def __len__(self) -> int:
        return self.num_graphs

    @property
    def params(self) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        return self.node_features, self.edge_features, self.edge_index, self.graph_index

    def process(self) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        # Get node and edge features, as well as edge indexes
        node_features: List[List[bool]] = [get_atom_features(atom) for atom in self.mol.GetAtoms()]
        edge_features: List[List[bool]] = [get_bond_features(bond) for bond in self.mol.GetBonds()]
        edge_index: List[List[int]] = [
            [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()] for bond in self.mol.GetBonds()
        ]

        # Make them tensors
        node_features = torch.tensor(node_features, dtype=torch.float)
        edge_features = torch.tensor(edge_features, dtype=torch.float)
        edge_index = torch.tensor(edge_index, dtype=torch.long).T

        # Make edges bidirectional
        edge_features = edge_features.repeat(2, 1)
        edge_index = torch.cat([edge_index, edge_index.flip(0)], 1)

        # Graph index
        graph_index = torch.zeros((node_features.size(0),), dtype=torch.long)

        return node_features, edge_features, edge_index, graph_index

    @property
    def num_nodes(self) -> int:
        return self.node_features.size(0)

    @property
    def num_edges(self) -> int:
        return self.edge_features.size(0)

    @property
    def num_graphs(self) -> int:
        return 1

    @property
    def num_node_features(self) -> int:
        return self.node_features.size(1)

    @property
    def num_edge_features(self) -> int:
        return self.edge_features.size(1)

    @property
    def device(self):
        devices = list(set([p.device for p in self.params]))
        assert len(devices) == 1
        return devices[0]

    def to(self, device):
        self.node_features = self.node_features.to(device)
        self.edge_features = self.edge_features.to(device)
        self.edge_index = self.edge_index.to(device)
        self.graph_index = self.graph_index.to(device)
        return self


class PackedTorchMol:
    def __init__(self, torch_mols: List[TorchMol]):
        super().__init__()
        self.torch_mols: List[TorchMol] = torch_mols
        self.mols: List[Mol] = [torch_mol.mol for torch_mol in self.torch_mols]
        self.node_features, self.edge_features, self.edge_index, self.graph_index = self.process()

    def __len__(self) -> int:
        return self.num_graphs

    @property
    def params(self) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        return self.node_features, self.edge_features, self.edge_index, self.graph_index

    def process(self) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        num_nodes = [tm.num_nodes for tm in self.torch_mols]
        cum_num_nodes = torch.tensor([0] + num_nodes).cumsum(0)[:-1]
        node_features = torch.cat([tm.node_features for tm in self.torch_mols], 0)
        edge_index = torch.cat(
            [tm.edge_index + i for i, tm in zip(cum_num_nodes, self.torch_mols)], 1,
        )
        edge_features = torch.cat([tm.edge_features for tm in self.torch_mols], 0)
        graph_index = torch.cat(
            [i + torch.zeros_like(tm.graph_index) for i, tm in enumerate(self.torch_mols)], 0,
        )
        return node_features, edge_features, edge_index, graph_index

    @property
    def num_nodes(self) -> int:
        return self.node_features.size(0)

    @property
    def num_edges(self) -> int:
        return self.edge_features.size(0)

    @property
    def num_graphs(self) -> int:
        return len(self.torch_mols)

    @property
    def num_node_features(self) -> int:
        return self.node_features.size(1)

    @property
    def num_edge_features(self) -> int:
        return self.edge_features.size(1)

    @property
    def device(self):
        devices = list(set([p.device for p in self.params]))
        assert len(devices) == 1
        return devices[0]

    def to(self, device):
        self.node_features = self.node_features.to(device)
        self.edge_features = self.edge_features.to(device)
        self.edge_index = self.edge_index.to(device)
        self.graph_index = self.graph_index.to(device)
        return self
