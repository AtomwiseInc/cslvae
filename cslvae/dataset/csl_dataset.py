import numpy as np
import pandas as pd
import sys
import torch
from rdkit import Chem
from rdkit.Chem.AllChem import ReactionFromSmarts
from torch.utils.data import Dataset
from typing import Iterable, Iterator, List, Optional, Tuple

from cslvae.utils.other_utils import flatten_list, int2mix
from cslvae.utils.torch_utils import build_library_indexes, convert_1d_to_2d_indexes
from cslvae.data import TorchMol, PackedTorchMol, NUM_NODE_FEATURES, NUM_EDGE_FEATURES


class CSLDataset(Dataset):
    def __init__(self, reaction_smarts_path: str, synthon_smiles_path: str):
        # Load reaction SMARTS and synthon SMILES dataframes
        reaction_df = pd.read_csv(reaction_smarts_path, sep="\s+")
        synthon_df = pd.read_csv(synthon_smiles_path, sep="\s+")

        # Map the original reaction_ids to the internal (normalized) reaction_ids
        reaction_df["orig_reaction_id"] = reaction_df["reaction_id"]
        reaction_df["reaction_smarts"] = reaction_df["smarts"]
        orig_reaction_mapper = {name: i for i, name in enumerate(reaction_df["reaction_id"])}
        reaction_df["reaction_id"] = (
            reaction_df["orig_reaction_id"].apply(lambda x: orig_reaction_mapper[x])
        )

        # Map the original synthon_ids to the internal (normalized) synthon_ids
        synthon_df["orig_synthon_id"] = synthon_df["synthon_id"]
        synthon_df["synthon_smiles"] = synthon_df["smiles"]
        synthon_mapper = {
            smi: i for i, smi in enumerate(sorted(synthon_df["synthon_smiles"].unique()))
        }
        synthon_df["synthon_id"] = synthon_df["synthon_smiles"].apply(lambda x: synthon_mapper[x])
        synthon_df["reaction_id"] = synthon_df["reaction_id"].apply(
            lambda x: orig_reaction_mapper[x]
        )

        # Form dicts for mapping internal reaction/synthon IDs to originals
        self._orig_reaction_id_lookup = {
            reaction_id: orig_reaction_id
            for reaction_id, orig_reaction_id in zip(
                reaction_df.reaction_id,
                reaction_df.orig_reaction_id,
            )
        }
        self._orig_synthon_id_lookup = {
            (reaction_id, synthon_id): orig_synthon_id
            for reaction_id, synthon_id, orig_synthon_id in zip(
                synthon_df.reaction_id,
                synthon_df.synthon_id,
                synthon_df.orig_synthon_id,
            )
        }

        # Form libtree; this is a list-of-list-of-list-of-ints. The top level corresponds to
        # reactions (i.e., len(libtree) == n_reactions), with the subsequent level corresponding
        # to the reaction R-groups, and the leaves are the normalized synthon_ids (ints) for the
        # synthons contained in the given R-group
        libtree = synthon_df[["reaction_id", "rgroup", "synthon_id"]]
        libtree = libtree.drop_duplicates()
        libtree = (
            libtree.sort_values(by=["reaction_id", "rgroup", "synthon_id"]).reset_index(drop=True)
        )
        libtree = libtree.groupby("reaction_id")
        libtree = libtree.apply(
            lambda x: x.groupby("rgroup").apply(lambda x: x["synthon_id"].tolist())
        ).T
        libtree: List[List[List[int]]] = [
            libtree[i].tolist() for i in range(len(orig_reaction_mapper))
        ]

        assert len(set(orig_reaction_mapper.values())) == len(set(orig_reaction_mapper.keys()))
        tmp = synthon_df[["orig_synthon_id", "synthon_smiles"]].drop_duplicates()
        assert len(tmp.orig_synthon_id.unique()) == len(tmp)
        orig_synthon_mapper = {
            k: synthon_mapper[v] for k, v in zip(tmp.orig_synthon_id, tmp.synthon_smiles)
        }

        # Retain only specific columns
        reaction_df = reaction_df[
            ["reaction_id", "orig_reaction_id", "reaction_smarts"]
        ].sort_values(by="reaction_id")
        synthon_df = synthon_df[
            ["synthon_id", "orig_synthon_id", "synthon_smiles"]
        ].sort_values(by="synthon_id")

        # Create a bunch of attributes that will be utilized by CSLDataset's methods
        self.reaction_df: pd.DataFrame = reaction_df.drop_duplicates().reset_index(drop=True)
        self.synthon_df: pd.DataFrame = synthon_df.drop_duplicates().reset_index(drop=True)
        self.libtree: List[List[List[int]]] = libtree
        self.reaction_smarts: List[str] = (
            self.reaction_df[["reaction_id", "reaction_smarts"]]
            .drop_duplicates()["reaction_smarts"].tolist()
        )
        self.synthon_smiles: List[str] = (
            self.synthon_df[["synthon_id", "synthon_smiles"]]
            .drop_duplicates()["synthon_smiles"].tolist()
        )
        self._rgroup_counts = [[len(x) for x in v] for k, v in enumerate(self.libtree)]
        self._reaction_counts = np.array(
            [np.prod(self._rgroup_counts[k]) for k in range(self.num_reactions)]
        )
        self._reaction_counts_cum = np.insert(np.cumsum(self._reaction_counts), 0, 0)
        self._orig_synthon_mapper = orig_synthon_mapper
        self._orig_reaction_mapper = orig_reaction_mapper
        self._num_products = sum([np.prod([len(y) for y in x]) for x in self.libtree])
        self._num_rgroups = sum([len(x) for x in self._rgroup_counts])

    def get_internal_synthon_id(self, orig_synthon_id) -> int:
        return self._orig_synthon_mapper[orig_synthon_id]

    def get_internal_reaction_id(self, orig_reaction_id) -> int:
        return self._orig_reaction_mapper[orig_reaction_id]

    @property
    def num_node_features(self) -> int:
        return NUM_NODE_FEATURES

    @property
    def num_edge_features(self) -> int:
        return NUM_EDGE_FEATURES

    @property
    def num_reactions(self) -> int:
        return len(self.reaction_smarts)

    @property
    def num_rgroups(self) -> int:
        return self._num_rgroups

    @property
    def num_synthons(self) -> int:
        return len(self.synthon_smiles)

    @property
    def num_products(self) -> int:
        return self._num_products

    def get_product_ids_by_reaction_id(self, reaction_id: int) -> Iterable[int]:
        return range(
            self._reaction_counts_cum[reaction_id], self._reaction_counts_cum[reaction_id + 1]
        )

    def product2smiles(self, reaction_id: int, synthon_ids: Tuple[int, ...]) -> str:
        return Chem.MolToSmiles(self.product2mol(reaction_id, synthon_ids))

    def product2mol(self, reaction_id: int, synthon_ids: Tuple[int, ...]) -> Chem.rdchem.Mol:
        reaction = self.reaction2rxn(reaction_id)
        synthons = tuple(self.synthon2mol(i) for i in synthon_ids)
        product = reaction.RunReactants(synthons)[0][0]
        return product

    def product2key(
        self, reaction_id: int, synthon_ids: Tuple[int, ...],
    ) -> Tuple[str, Tuple[str, ...]]:
        orig_reaction_id = self._orig_reaction_id_lookup[reaction_id]
        orig_synthon_ids = []
        for synthon_id in synthon_ids:
            orig_synthon_id = self._orig_synthon_id_lookup[
                (reaction_id, synthon_id)
            ]
            orig_synthon_ids.append(str(orig_synthon_id))
        return str(orig_reaction_id), tuple(orig_synthon_ids)

    def reaction2smarts(self, reaction_id: int) -> str:
        return self.reaction_smarts[reaction_id]

    def reaction2rxn(self, reaction_id: int) -> Chem.rdChemReactions.ChemicalReaction:
        return ReactionFromSmarts(self.reaction2smarts(reaction_id))

    def synthon2smiles(self, synthon_id: int) -> str:
        return self.synthon_smiles[synthon_id]

    def synthon2mol(self, synthon_id: int) -> Chem.rdchem.Mol:
        return Chem.MolFromSmiles(self.synthon2smiles(synthon_id))

    def __len__(self) -> int:
        return self.num_products

    def __getitem__(self, product_id: int):
        try:
            assert 0 <= product_id < len(self)
            reaction_id = (self._reaction_counts_cum <= product_id).sum() - 1
            mix = int2mix(
                product_id - self._reaction_counts_cum[reaction_id],
                self._rgroup_counts[reaction_id],
            )
            synthon_ids = tuple(x[i] for i, x in zip(mix, self.libtree[reaction_id]))
            return {
                "product_id": product_id,
                "reaction_id": reaction_id,
                "synthon_ids": synthon_ids,
                "product": TorchMol(self.product2smiles(reaction_id, synthon_ids)),
                "synthons": [TorchMol(self.synthon2smiles(i)) for i in synthon_ids],
            }
        except:
            return None

    def create_dataloader(
        self,
        products_per_reaction: int,
        num_reactions: int = 1,
        max_iterations: Optional[int] = None,
    ) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            dataset=self,
            batch_sampler=self.create_batch_sampler(
                products_per_reaction, num_reactions, max_iterations,
            ),
            collate_fn=self.collate_fn,
        )

    def create_batch_sampler(
        self,
        products_per_reaction: int,
        num_reactions: int = 1,
        max_iterations: Optional[int] = None,
    ):
        class ReactionBatchSampler:
            def __init__(self_, dataset, products_per_reaction, num_reactions, max_iterations):
                self_.dataset = dataset
                self_.products_per_reaction = int(products_per_reaction)
                self_.num_reactions = int(num_reactions)
                self_.max_iterations = int(max_iterations or sys.maxsize)
                self_.cum_iterations = 0

            def __iter__(self_) -> Iterator[List[int]]:
                self_.cum_iterations = 0
                while self_.cum_iterations < self_.max_iterations:
                    self_.cum_iterations += 1
                    reaction_ids = (
                        np.random.choice(
                            range(self_.dataset.num_reactions), self_.num_reactions, replace=False,
                        )
                        .tolist()
                    )
                    ranges = [
                        self_.dataset.get_product_ids_by_reaction_id(i) for i in reaction_ids
                    ]
                    fn = (
                        lambda rng: np.random.randint(
                            rng.start, rng.stop, (self_.products_per_reaction,)
                        )
                    )
                    indexes = flatten_list([fn(rng).tolist() for rng in ranges])
                    yield indexes

        return ReactionBatchSampler(self, products_per_reaction, num_reactions, max_iterations)

    @staticmethod
    def collate_fn(items):
        items = [item for item in items if item is not None]
        libtree = {}
        for item in items:
            (reaction_id, synthon_ids) = (item["reaction_id"], item["synthon_ids"])
            if reaction_id not in libtree:
                libtree[reaction_id] = [{s} for s in synthon_ids]
            else:
                for i in range(len(libtree[reaction_id])):
                    libtree[reaction_id][i].update({synthon_ids[i]})

        orig_reaction_ids = sorted(libtree.keys())
        libtree = [list(map(sorted, libtree[k])) for k in orig_reaction_ids]

        library_indexes = build_library_indexes(libtree)

        reaction_mapper = {j: i for i, j in enumerate(orig_reaction_ids)}
        synthon_mapper = {j.item(): i for i, j in enumerate(library_indexes["orig_synthon_ids"])}

        product2reaction: List[int] = [reaction_mapper[item["reaction_id"]] for item in items]
        block2product: List[int] = flatten_list(
            [len(item["synthon_ids"]) * [i] for i, item in enumerate(items)]
        )
        block2rgroup: List[int] = flatten_list(
            [list(range(len(item["synthon_ids"]))) for i, item in enumerate(items)]
        )
        block2synthon: List[int] = flatten_list(
            [[synthon_mapper[i] for i in item["synthon_ids"]] for item in items]
        )

        product2reaction = convert_1d_to_2d_indexes(torch.tensor(product2reaction))
        block2product = convert_1d_to_2d_indexes(torch.tensor(block2product))
        block2reaction = convert_1d_to_2d_indexes(product2reaction[1][block2product[1]])
        block2rgroup = convert_1d_to_2d_indexes(
            torch.tensor(block2rgroup) + 
            library_indexes["first_rgroup_by_reaction"][block2reaction[1]]
        )
        block2synthon = convert_1d_to_2d_indexes(torch.tensor(block2synthon))

        (idx0, idx1) = block2synthon[:, block2synthon[1].argsort()]
        idx2 = idx0[torch.where(torch.nn.functional.pad(idx1.diff(), (1, 0), value=1))[0]]

        products = [item["product"] for item in items]
        blocks = flatten_list([item["synthons"] for item in items])
        synthons = [blocks[i.item()] for i in idx2]

        return {
            "library_indexes": library_indexes,
            "product2reaction": product2reaction,
            "block2product": block2product,
            "block2reaction": block2reaction,
            "block2rgroup": block2rgroup,
            "block2synthon": block2synthon,
            "products": PackedTorchMol(products),
            "synthons": PackedTorchMol(synthons),
        }
