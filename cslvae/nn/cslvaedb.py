import os
import time
import torch
from functools import partial
from torch import nn, Tensor
from torch_scatter import scatter_max
from typing import List, Tuple

from cslvae.data import PackedTorchMol, TorchMol
from cslvae.dataset import CSLDataset, SmilesDataset
from cslvae.nn import CSLVAE
from cslvae.utils.torch_utils import add_gumbel_like, build_library_indexes


class CSLVAEDB(nn.Module):
    def __init__(self, cslvae: CSLVAE, dataset: CSLDataset):
        super().__init__()
        self.cslvae = cslvae
        self.dataset = dataset

        device = next(self.cslvae.parameters()).device
        make_zeros = partial(torch.zeros, dtype=torch.float, device=device)
        synthon_keys = make_zeros(self.dataset.num_synthons, self.cslvae.synthon_key_dim)
        reaction_keys = make_zeros(self.dataset.num_reactions, self.cslvae.reaction_key_dim)
        rgroup_feats = make_zeros(self.dataset.num_rgroups, self.cslvae.query_dim)
        reaction_feats = make_zeros(self.dataset.num_reactions, self.cslvae.query_dim)
        self.library_tensors = nn.ParameterDict(
            {
                "synthon_keys": nn.Parameter(synthon_keys, False),
                "reaction_keys": nn.Parameter(reaction_keys, False),
                "rgroup_feats": nn.Parameter(rgroup_feats, False),
                "reaction_feats": nn.Parameter(reaction_feats, False),
            }
        )
        self.library_indexes = nn.ParameterDict(
            {
                k: nn.Parameter(v, requires_grad=False)
                for k, v in build_library_indexes(self.dataset.libtree, device).items()
            }
        )

    @torch.no_grad()
    def encode(self, smiles_list: List[str]) -> Tensor:
        self.eval()
        device = next(self.cslvae.parameters()).device
        molecules = PackedTorchMol([TorchMol(s) for s in smiles_list])
        queries = self.cslvae.encode_molecules(molecules.to(device))
        return queries

    @torch.no_grad()
    def decode(
        self,
        queries: Tensor,
        reaction_temperature: float = 1.0,
        synthon_temperature: float = 1.0,
    ) -> List[Tuple[str, Tuple[str, Tuple[str, ...]]]]:
        # Set to eval mode
        self.eval()

        # Select one reaction per molecule via sampling or argmax decision rule
        reaction_logits = self.cslvae.get_reaction_logits(queries, self.library_tensors)
        if reaction_temperature > 0:
            reactions = add_gumbel_like(reaction_logits / reaction_temperature).max(1).indices
        else:
            reactions = reaction_logits.max(1).indices

        # Select one synthon per R-group position (block) via sampling or argmax
        # decision rule
        (block2synthon_logits, block2synthon_choices) = (
            self.cslvae.get_synthon_logits(
                queries, self.library_tensors, self.library_indexes, reactions,
            )
        )
        if synthon_temperature > 0:
            _, synthon_idx = scatter_max(
                add_gumbel_like(block2synthon_logits / synthon_temperature),
                block2synthon_choices[0],
                0,
            )
        else:
            _, synthon_idx = scatter_max(block2synthon_logits, block2synthon_choices[0], 0)
        synthons = block2synthon_choices[1][synthon_idx]

        # Decode the products and subsequently the keys
        idx = torch.nn.functional.pad(
            self.library_indexes["n_rgroups_by_reaction"][reactions].cumsum(0), (1, 0),
        )
        products = [
            (reactions[i].item(), tuple(j.item() for j in synthons[start:end]))
            for i, (start, end) in enumerate(zip(idx[:-1], idx[1:]))
        ]
        return [
            (self.dataset.product2smiles(*product), self.dataset.product2key(*product))
            for product in products
        ]

    def fit(self, config: dict) -> None:
        """
        Represent a CSLDataset with CSLVAE
        """
        batch_size = int(config.get("batch_size"))
        num_workers = int(config.get("num_workers", os.cpu_count()))
        pin_memory = bool(config.get("pin_memory", False))

        synthons_dataset = SmilesDataset(smiles_list=self.dataset.synthon_smiles)
        num_synthons = len(synthons_dataset)
        max_iters = num_synthons // batch_size + (num_synthons % batch_size > 0)
        dataloader = torch.utils.data.DataLoader(
            dataset=synthons_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=synthons_dataset.collate_fn,
            shuffle=False,
        )

        device = next(self.cslvae.parameters()).device
        print(f"Encoding {num_synthons:,} synthons.")
        self.cslvae.eval()
        synthon_feats = torch.zeros(
            (self.dataset.num_synthons, self.cslvae.query_dim), dtype=torch.float, device=device,
        )
        with torch.no_grad():
            start_time = time.time()
            for batch_iter, batch in enumerate(dataloader):
                synthons = batch.get("molecules").to(device)
                idx = batch.get("idx").to(device)
                synthon_feats[idx] = self.cslvae.encode_synthons(synthons)

            print(f"Synthon encoding complete after {time.time() - start_time:.2f} seconds.")

            library_tensors = self.cslvae.encode_library(synthon_feats, self.library_indexes)

            self.library_tensors.synthon_keys[:] = library_tensors["synthon_keys"]
            self.library_tensors.rgroup_feats[:] = library_tensors["rgroup_feats"]
            self.library_tensors.reaction_feats[:] = library_tensors["reaction_feats"]
            self.library_tensors.reaction_keys[:] = library_tensors["reaction_keys"]

            del synthon_feats, library_tensors

            print(f"Library encoding complete.")
