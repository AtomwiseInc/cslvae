from rdkit.Chem.rdchem import Atom, Bond
from typing import List, Tuple


def _get_return(i, types) -> Tuple[int, int]:
    return {i: j for j, i in enumerate(types)}[i], len(types)


def _one_hot_featurizer(idx: int, length: int) -> List[bool]:
    one_hot = [False] * length
    one_hot[idx] = True
    return one_hot


def get_default_allowed_elements() -> List[str]:
    return ["*", "B", "Br", "C", "Cl", "F", "Fe", "I", "N", "O", "P", "S", "Se", "Si", "Sn"]


def get_default_sticky_elements() -> List[str]:
    return ["*", "U", "Np", "Pu", "Am"]


def reassign_sticky_elements(element: str) -> str:
    sticky_elements = get_default_sticky_elements()
    if element in sticky_elements:
        return "*"
    else:
        return element


def get_atom_element_idx(atom: Atom) -> Tuple[int, int]:
    i = reassign_sticky_elements(atom.GetSymbol())
    types = get_default_allowed_elements()
    return _get_return(i, types)


def get_atom_degree_idx(atom: Atom) -> Tuple[int, int]:
    i = min(max(1, atom.GetDegree()), 7)
    types = list(range(1, 8))
    return _get_return(i, types)


def get_atom_hybridization_idx(atom: Atom) -> Tuple[int, int]:
    i = atom.GetHybridization().name
    types = ["S", "SP", "SP2", "SP3", "SP3D", "SP3D2", "UNSPECIFIED"]
    return _get_return(i, types)


def get_atom_chirality_idx(atom: Atom) -> Tuple[int, int]:
    i = atom.GetChiralTag().name
    types = ["CHI_UNSPECIFIED", "CHI_OTHER", "CHI_TETRAHEDRAL_CW", "CHI_TETRAHEDRAL_CCW"]
    return _get_return(i, types)


def get_atom_hydrogens_idx(atom: Atom) -> Tuple[int, int]:
    i = min(atom.GetTotalNumHs(), 7)
    types = list(range(8))
    return _get_return(i, types)


def get_atom_charge_idx(atom: Atom) -> Tuple[int, int]:
    i = min(max(-3, atom.GetFormalCharge()), 3)
    types = list(range(-3, 4))
    return _get_return(i, types)


def get_atom_aromaticity_idx(atom: Atom) -> Tuple[int, int]:
    i = atom.GetIsAromatic()
    types = [False, True]
    return _get_return(i, types)


def get_bond_type_idx(bond: Bond) -> Tuple[int, int]:
    i = bond.GetBondType().name
    types = ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]
    return _get_return(i, types)


def get_bond_conjugated_idx(bond: Bond) -> Tuple[int, int]:
    i = bond.GetIsConjugated()
    types = [False, True]
    return _get_return(i, types)


def get_bond_ring_idx(bond: Bond) -> Tuple[int, int]:
    i = bond.IsInRing()
    types = [False, True]
    return _get_return(i, types)


def get_bond_stereo_idx(bond: Bond) -> Tuple[int, int]:
    i = bond.GetStereo().name
    types = ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"]
    return _get_return(i, types)


def get_atom_features(atom: Atom) -> List[bool]:
    func_order = [
        get_atom_element_idx,
        get_atom_degree_idx,
        get_atom_hybridization_idx,
        get_atom_chirality_idx,
        get_atom_hydrogens_idx,
        get_atom_charge_idx,
        get_atom_aromaticity_idx,
    ]
    out = []
    for func in func_order:
        idx, length = func(atom)
        out.extend(_one_hot_featurizer(idx, length))
    return out


def get_bond_features(bond: Bond) -> List[bool]:
    func_order = [
        get_bond_type_idx,
        get_bond_conjugated_idx,
        get_bond_ring_idx,
        get_bond_stereo_idx,
    ]
    out = []
    for func in func_order:
        idx, length = func(bond)
        out.extend(_one_hot_featurizer(idx, length))
    return out
