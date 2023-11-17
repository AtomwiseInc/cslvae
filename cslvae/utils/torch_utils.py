import copy
import torch
from torch import LongTensor, Tensor
from typing import Dict, List, Optional

from .other_utils import flatten_list


def gumbel_like(x: Tensor) -> Tensor:
    """
    Samples a standard Gumbel random variable with the same shape as x.
    """
    return (-torch.rand_like(x).log()).log()


def add_gumbel_like(x: Tensor) -> Tensor:
    """
    Samples a standard Gumbel random variable with the same shape as x and adds it to x.
    """
    return x + gumbel_like(x)


def batched_lookup(queries: Tensor, keys: Tensor, max_comparisons: Optional[int] = None) -> Tensor:
    """
    Returns lookup scores given a batch of query-key pairs.
    Input:
        queries (m, k)
        keys (m, k)
    Output:
        scores (m, )
    """
    assert queries.ndim == keys.ndim == 2
    assert queries.shape == keys.shape
    batch_size, dim = queries.shape
    if max_comparisons is None:
        max_comparisons = batch_size
    assert isinstance(max_comparisons, int) and max_comparisons > 0
    (queries, keys) = (queries.unsqueeze(1), keys.unsqueeze(2))
    (start_idx, end_idx) = (0, max_comparisons)
    dotprods = []
    while start_idx < batch_size:
        dotprods.append(torch.bmm(queries[start_idx:end_idx], keys[start_idx:end_idx]))
        (start_idx, end_idx) = (end_idx, end_idx + max_comparisons)
    return torch.cat(dotprods, 0).squeeze(-1).squeeze(-1) / (dim ** 0.5)


def torch_lexsort(x: Tensor) -> Tensor:
    assert x.ndim == 2
    _, inv = torch.unique(x, dim=-1, sorted=True, return_inverse=True)
    idx = torch.argsort(inv)
    return x[:, idx]


def convert_1d_to_2d_indexes(indexes: LongTensor) -> LongTensor:
    assert indexes.ndim == 1
    return torch.stack([torch.arange(indexes.size(0), dtype=torch.long, device=indexes.device), indexes], 0)


def build_library_indexes(libtree: List[List[List[int]]], device: str = "cpu") -> Dict[str, LongTensor]:
    # Prepare synthons
    orig_synthons: List[int] = sorted(set(flatten_list(flatten_list(libtree))))

    synthons_mapper = {k: i for i, k in enumerate(orig_synthons)}
    synthons_mapper_reverse = {i: k for k, i in synthons_mapper.items()}

    # Renumber the synthon IDs
    libtree = copy.deepcopy(libtree)
    for i, ix in enumerate(libtree):
        for j, jx in enumerate(ix):
            libtree[i][j] = list(map(lambda k: synthons_mapper[k], jx))

    # Form all mapping indexes
    libtree_1l: List[List[int]] = flatten_list(libtree)
    libtree_2l: List[int] = flatten_list(libtree_1l)
    synthons: List[int] = sorted(set(libtree_2l))

    orig_synthon_ids = torch.tensor([synthons_mapper_reverse[i] for i in synthons], dtype=torch.long)
    n_reactions = torch.tensor(len(libtree), dtype=torch.long)
    n_rgroups = torch.tensor(len(libtree_1l), dtype=torch.long)
    n_synthons = torch.tensor(len(synthons), dtype=torch.long)

    synthon2rgroup_by_reaction = [list(map(lambda x: (x, i), idx)) for i, idx in enumerate(libtree_1l)]

    n_rgroups_by_reaction = torch.tensor([len(idx) for idx in libtree], dtype=torch.long)
    n_synthons_by_rgroup = torch.tensor([len(idx) for idx in libtree_1l], dtype=torch.long)

    first_rgroup_by_reaction = torch.nn.functional.pad(n_rgroups_by_reaction, (1, 0)).cumsum(0)[:-1]
    synthon2rgroup = torch_lexsort(torch.tensor(flatten_list(synthon2rgroup_by_reaction), dtype=torch.long).T)

    rgroup2reaction = convert_1d_to_2d_indexes(
        torch.arange(n_reactions, dtype=torch.long).repeat_interleave(n_rgroups_by_reaction)
    )

    synthons_by_rgroup = [torch.tensor(idx, dtype=torch.long) for idx in libtree_1l]

    library_indexes = {
        "n_reactions": n_reactions,
        "n_rgroups": n_rgroups,
        "n_synthons": n_synthons,
        "orig_synthon_ids": orig_synthon_ids,
        "synthon2rgroup": synthon2rgroup,
        "rgroup2reaction": rgroup2reaction,
        "n_rgroups_by_reaction": n_rgroups_by_reaction,
        "n_synthons_by_rgroup": n_synthons_by_rgroup,
        "first_rgroup_by_reaction": first_rgroup_by_reaction,
        **{f"synthons_where_rgroup_{i}": idx for i, idx in enumerate(synthons_by_rgroup)},
    }
    return {k: v.to(device) for k, v in library_indexes.items()}
