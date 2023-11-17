import os
import torch
from torch import distributions as D, nn, tensor, LongTensor, Tensor
from torch.utils.tensorboard import SummaryWriter
from torch_scatter import scatter_log_softmax, scatter_max, scatter_mean, scatter_min, scatter_sum
from typing import Dict, Optional, Tuple, Union

from cslvae.data import NUM_NODE_FEATURES, NUM_EDGE_FEATURES, PackedTorchMol, TorchMol
from cslvae.dataset import CSLDataset
from cslvae.utils.torch_utils import batched_lookup
from .mpnn import EdgeMessagePassingNetwork
from .scatter import ModularizedScatter


class CSLVAE(nn.Module):
    def __init__(
            self,
            query_dim: int,
            node_dim: int,
            edge_dim: int,
            hidden_dim: int,
            reaction_key_dim: int,
            synthon_key_dim: int,
            num_layers: int,
            residual: bool = True,
            normalize: bool = True,
            aggregate: str = "sum",
    ):
        super().__init__()
        self.query_dim = int(query_dim)
        self.node_dim = int(node_dim)
        self.edge_dim = int(edge_dim)
        self.hidden_dim = int(hidden_dim)
        self.reaction_key_dim = int(reaction_key_dim)
        self.synthon_key_dim = int(synthon_key_dim)
        self.num_layers = int(num_layers)
        self.residual = bool(residual)
        self.normalize = bool(normalize)
        self.aggregate = str(aggregate)

        # Define modules
        self.atom_embedder = nn.Linear(NUM_NODE_FEATURES, self.node_dim)
        self.bond_embedder = nn.Linear(NUM_EDGE_FEATURES, self.edge_dim)
        self.molecular_encoder = EdgeMessagePassingNetwork(
            node_dim=self.node_dim,
            edge_dim=self.edge_dim,
            graph_dim=self.query_dim,
            hidden_dim=self.hidden_dim,
            residual=self.residual,
            normalize=self.normalize,
            aggregate=self.aggregate,
            num_layers=self.num_layers,
        )
        self.molecular_inference_net = nn.Linear(self.query_dim, self.query_dim)
        self.molecular_processor_net = nn.Sequential(
            nn.Linear(self.query_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.query_dim),
        )
        self.synthon_encoder = EdgeMessagePassingNetwork(
            node_dim=self.node_dim,
            edge_dim=self.edge_dim,
            graph_dim=self.query_dim,
            hidden_dim=self.hidden_dim,
            residual=self.residual,
            normalize=self.normalize,
            aggregate=self.aggregate,
            num_layers=self.num_layers,
        )
        self.rgroup_encoder_p0 = nn.Sequential(
            nn.Linear(self.query_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.rgroup_encoder_p1 = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.query_dim),
        )
        self.rgroup_encoder_pooling_function = ModularizedScatter(scatter_op=scatter_mean)
        self.reaction_encoder_p0 = nn.Sequential(
            nn.Linear(self.query_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.reaction_encoder_p1 = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.query_dim),
        )
        self.reaction_encoder_pooling_function = ModularizedScatter(scatter_op=scatter_sum)
        self.reaction_key_generator = nn.Linear(self.query_dim, self.reaction_key_dim)
        self.synthon_key_generator = nn.Linear(self.query_dim, self.synthon_key_dim)
        self.reaction_query_generator = nn.Sequential(
            nn.Linear(self.query_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.reaction_key_dim),
        )
        self.synthon_query_generator = nn.Sequential(
            nn.Linear(self.query_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.synthon_key_dim),
        )

    def fit(self, dataset: CSLDataset, config: dict, outdir: Optional[str] = None) -> None:
        # Get devive
        device = next(self.parameters()).device

        # Unpack config
        reactions_per_batch = int(config.get("reactions_per_batch"))
        products_per_reaction = int(config.get("products_per_reaction"))
        logging_iterations = int(config.get("logging_iterations"))
        checkpoint_iterations = int(config.get("checkpoint_iterations", 0))
        max_iterations = int(config.get("max_iterations"))
        beta_start = float(config.get("beta_start"))
        beta_stop = float(config.get("beta_stop"))
        beta_step = float(config.get("beta_step"))
        beta_step_iterations = int(config.get("beta_step_iterations"))
        assert (beta_start <= beta_stop) and (beta_step >= 0)

        # Instantiate writer
        writer = None if outdir is None else SummaryWriter(log_dir=outdir)

        # Instantiate optimizer
        opt_name = str(config.get("optimizer_name"))
        opt_kwargs = dict(config.get("optimizer_kwargs"))
        opt_cls = getattr(torch.optim, opt_name)
        optimizer = opt_cls(self.parameters(), **opt_kwargs)

        # Instantiate the dataloader and begin training
        dataloader = dataset.create_dataloader(
            products_per_reaction, reactions_per_batch, max_iterations,
        )
        beta = beta_start - beta_step
        print(f"Beginning training for {max_iterations:,} iterations.")
        self.train()
        for iteration, batch in enumerate(dataloader):

            # Update beta every `beta_step_iterations` iterations
            if iteration % beta_step_iterations == 0:
                beta = min(beta_stop, beta + beta_step)

            # Reset gradient
            optimizer.zero_grad()

            # Unpack batch items
            library_indexes: Dict[str, LongTensor] = {
                k: v.to(device) for k, v in batch["library_indexes"].items()
            }
            product2reaction: LongTensor = batch["product2reaction"].to(device)
            block2product: LongTensor = batch["block2product"].to(device)
            block2synthon: LongTensor = batch["block2synthon"].to(device)
            synthons: Union[TorchMol, PackedTorchMol] = batch["synthons"].to(device)
            products: Union[TorchMol, PackedTorchMol] = batch["products"].to(device)

            # Encode the synthons
            synthon_feats: Tensor = self.encode_synthons(synthons)

            # Encode the library
            library_tensors: Dict[str, Tensor] = self.encode_library(
                synthon_feats, library_indexes,
            )

            # Encode the products
            queries: Tensor = self.encode_molecules(products)

            # Get reaction logits
            product2reaction_logits: Tensor = self.get_reaction_logits(queries, library_tensors)

            # Get R-group assignment logits; uses teacher forcing
            (block2synthon_logits, block2synthon_choices) = self.get_synthon_logits(
                queries, library_tensors, library_indexes, product2reaction[1],
            )
            _, block2synthon_labels = torch.where(
                (block2synthon_choices.unsqueeze(1) - block2synthon.unsqueeze(2) == 0).all(0)
            )

            # Get negative log likelihood components
            product2reaction_logp = product2reaction_logits.log_softmax(1)
            product2reaction_nll = -product2reaction_logp[
                torch.arange(product2reaction.size(1), device=device), product2reaction[1]
            ]
            block2synthon_logp = scatter_log_softmax(
                block2synthon_logits, block2synthon_choices[0], 0,
            )
            block2synthon_nll = -scatter_sum(
                block2synthon_logp[block2synthon_labels], block2product[1], 0
            )

            # Get latent space prior and variational posterior 
            query_posterior = D.Normal(loc=queries.mean(0), scale=queries.std(0) + 1e-6)
            query_prior = D.Normal(
                loc=torch.zeros_like(query_posterior.loc),
                scale=torch.ones_like(query_posterior.scale),
            )

            # Calculate loss
            nll = product2reaction_nll + block2synthon_nll
            kld = D.kl_divergence(query_posterior, query_prior)
            loss = nll.mean() + beta * kld.mean() if beta > 0 else nll.mean()

            # Calculate accuracy based on argmax prediction rule
            product2reaction_pred = product2reaction_logits.max(1).indices
            block2synthon_pred = block2synthon_choices[1][
                scatter_max(block2synthon_logits, block2synthon_choices[0], 0)[1]
            ]
            product2reaction_correct = (product2reaction[1] == product2reaction_pred).float()
            block2synthon_correct = (block2synthon[1] == block2synthon_pred).float()
            correct = torch.minimum(
                product2reaction_correct, scatter_min(block2synthon_correct, block2product[1], 0)[
                    0
                ]
            )
            total_accuracy = correct.float().mean()

            # Form metrics dict
            metrics_dict = {
                "Loss": loss,
                "Reaction NLL": product2reaction_nll.mean(),
                "Synthon NLL": block2synthon_nll.mean(),
                "Negative log likelihood": nll.mean(),
                "KL divergence": kld.mean(),
                "Average likelihood": (-nll).exp().mean(),
                "Reaction top-1 accuracy": product2reaction_correct.mean(),
                "Synthon top-1 accuracy": block2synthon_correct.mean(),
                "Total top-1 accuracy": total_accuracy,
                "Beta": beta,
            }

            # Do backward pass and take optimizer step
            loss.backward()
            optimizer.step()

            if (iteration % logging_iterations == 0) or (iteration == max_iterations):
                print(
                    f"Iteration {iteration}. Loss {loss:.4f}. Top-1 accuracy {total_accuracy:.4f}."
                )
                nan_value = torch.zeros(()) / torch.zeros(())
                if writer is not None:
                    for k, v in metrics_dict.items():
                        writer.add_scalar(f"{k}", v if v is not None else nan_value, iteration)

            if outdir is not None and checkpoint_iterations > 0:
                if (iteration % checkpoint_iterations == 0) or (iteration == max_iterations):
                    checkpoint_path = os.path.join(
                        outdir, "checkpoints", f"checkpoint_{iteration}.pt"
                    )
                    print(f"Check-pointing model iteration {iteration} at {checkpoint_path}.")
                    state_dict = {
                        "iter": iteration,
                        "model_state_dict": self.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": loss,
                    }
                    torch.save(state_dict, checkpoint_path)

            if iteration == max_iterations:
                print(f"Maximum iterations ({max_iterations}) reached.")
                break

        if writer is not None:
            writer.close()

    def encode_synthons(self, synthons: Union[TorchMol, PackedTorchMol]) -> Tensor:
        device = next(self.parameters()).device
        synthons = synthons.to(device)
        _, _, synthon_feats = self.synthon_encoder(
            node_feats=self.atom_embedder(synthons.node_features),
            edge_feats=self.bond_embedder(synthons.edge_features),
            edge_index=synthons.edge_index,
            graph_feats=torch.zeros(
                (len(synthons), self.query_dim), device=device, dtype=torch.float
            ),
            graph_index=synthons.graph_index,
        )
        return synthon_feats

    def encode_library(
        self, synthon_feats: Tensor, library_indexes: Dict[str, LongTensor],
    ) -> Dict[str, Tensor]:
        # Move synthon features to same device as CSLVAE
        device = next(self.parameters()).device
        synthon_feats = synthon_feats.to(device)

        # Get library indexes
        synthon2rgroup = library_indexes["synthon2rgroup"]
        rgroup2reaction = library_indexes["rgroup2reaction"]

        # Encode library
        synthon_keys = self.synthon_key_generator(synthon_feats)
        rgroup_feats = self.rgroup_encoder_p1(
            self.rgroup_encoder_pooling_function(
                self.rgroup_encoder_p0(synthon_feats), synthon2rgroup,
            ),
        )
        reaction_feats = self.reaction_encoder_p1(
            self.reaction_encoder_pooling_function(
                self.reaction_encoder_p0(rgroup_feats), rgroup2reaction,
            ),
        )
        reaction_keys = self.reaction_key_generator(reaction_feats)
        return {
            "synthon_keys": synthon_keys,
            "rgroup_feats": rgroup_feats,
            "reaction_feats": reaction_feats,
            "reaction_keys": reaction_keys,
        }

    def encode_molecules(self, molecules: Union[TorchMol, PackedTorchMol]) -> Tensor:
        device = next(self.parameters()).device
        molecules = molecules.to(device)
        _, _, molecule_feats = self.molecular_encoder(
            node_feats=self.atom_embedder(molecules.node_features),
            edge_feats=self.bond_embedder(molecules.edge_features),
            edge_index=molecules.edge_index,
            graph_feats=torch.zeros(
                (len(molecules), self.query_dim), device=device, dtype=torch.float
            ),
            graph_index=molecules.graph_index,
        )
        queries = self.molecular_inference_net(molecule_feats)
        return queries

    def get_reaction_logits(self, queries: Tensor, library_tensors: Dict[str, Tensor]) -> Tensor:
        # Get device
        device = next(self.parameters()).device

        # For each molecular query, we start by constructing the reaction query
        queries = queries.to(device)
        molecule_feats = self.molecular_processor_net(queries)
        reaction_queries = self.reaction_query_generator(molecule_feats)

        # Select one reaction per molecule (via sampling or argmax decision rule)
        product2reaction_logits = (
            reaction_queries @ library_tensors["reaction_keys"].T / (self.reaction_key_dim ** 0.5)
        )
        return product2reaction_logits

    def get_synthon_logits(
        self,
        queries: Tensor,
        library_tensors: Dict[str, Tensor],
        library_indexes: Dict[str, LongTensor],
        reactions: LongTensor,
    ) -> Tuple[Tensor, LongTensor]:
        # Get device
        device = next(self.parameters()).device

        # Process queries into molecular features
        queries = queries.to(device)
        molecule_feats = self.molecular_processor_net(queries)

        # For each molecular query, we construct a synthon query in each R-group position ("block")
        # of the decoded reaction
        n_rgroups_by_product = library_indexes["n_rgroups_by_reaction"][reactions]
        idx0 = n_rgroups_by_product[:-1].cumsum(0)
        idx1 = torch.arange(n_rgroups_by_product.sum(), device=device)
        idx2 = torch.zeros_like(idx1)
        idx2[idx0] = n_rgroups_by_product[:-1]
        idx3 = idx1 - idx2.cumsum(0)
        block2rgroup = (
            library_indexes["first_rgroup_by_reaction"][
                reactions.repeat_interleave(n_rgroups_by_product)
            ] +
            idx3
        )
        n_blocks = block2rgroup.size(0)
        blocks = torch.arange(n_blocks, device=device)
        block_feats = (
            molecule_feats.repeat_interleave(n_rgroups_by_product, 0) +
            library_tensors["rgroup_feats"][block2rgroup] + 
            library_tensors["reaction_feats"][library_indexes["rgroup2reaction"][1][block2rgroup]]
        )
        synthon_queries = self.synthon_query_generator(block_feats)

        # Get all eligible synthons for each block
        idx0 = blocks.repeat_interleave(library_indexes["n_synthons_by_rgroup"][block2rgroup], 0)
        idx1 = torch.cat(
            [library_indexes[f"synthons_where_rgroup_{i.item()}"] for i in block2rgroup], 0,
        )
        block2synthon = torch.stack([idx0, idx1])

        # Select one synthon per block (through sampling or argmax decision rule)
        block2synthon_logits = batched_lookup(
            synthon_queries[block2synthon[0]], library_tensors["synthon_keys"][block2synthon[1]],
        )
        return block2synthon_logits, block2synthon
