from .torch_mol import TorchMol, PackedTorchMol

NUM_NODE_FEATURES = TorchMol("CC").num_node_features
NUM_EDGE_FEATURES = TorchMol("CC").num_edge_features
