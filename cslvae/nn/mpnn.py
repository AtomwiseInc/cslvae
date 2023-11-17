import torch
from torch import LongTensor, nn, Tensor
from torch_scatter import scatter_max, scatter_mean, scatter_sum
from typing import Optional, Tuple


class EdgeMessagePassingLayer(nn.Module):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        graph_dim: int,
        hidden_dim: int,
        drop_rate: float = 0.0,
        residual: bool = True,
        normalize: bool = True,
        aggregate: str = "sum",
    ):
        super().__init__()
        self.node_dim = int(node_dim)
        self.edge_dim = int(edge_dim)
        self.graph_dim = int(graph_dim)
        self.hidden_dim = int(hidden_dim)
        self.drop_rate = float(drop_rate)
        self.residual = bool(residual)
        self.normalize = bool(normalize)
        self.aggregate = str(aggregate)

        msg = "aggregate must be set to 'attn', 'max', 'mean', or 'sum'."
        assert self.aggregate in ["attn", "max", "mean", "sum"], msg
        use_attention = 1 if self.aggregate == "attn" else 0

        # Edge model
        self.edge_net = nn.ModuleDict()
        self.edge_net["node_row_fc0"] = nn.Linear(self.node_dim, self.hidden_dim)
        self.edge_net["node_col_fc0"] = nn.Linear(self.node_dim, self.hidden_dim, bias=False)
        self.edge_net["edge_fc0"] = nn.Linear(self.edge_dim, self.hidden_dim, bias=False)
        self.edge_net["graph_fc0"] = nn.Linear(self.graph_dim, self.hidden_dim, bias=False)
        self.edge_net["fc1"] = nn.Linear(self.hidden_dim, self.edge_dim + use_attention)
        if self.drop_rate > 0:
            self.edge_net["dropout"] = nn.Dropout(self.drop_rate)
        if self.normalize:
            self.edge_net["node_row_norm"] = nn.LayerNorm(self.node_dim)
            self.edge_net["node_col_norm"] = nn.LayerNorm(self.node_dim)
            self.edge_net["edge_norm"] = nn.LayerNorm(self.edge_dim)
            self.edge_net["graph_norm"] = nn.LayerNorm(self.graph_dim)
            self.edge_net["hidden_norm"] = nn.LayerNorm(self.hidden_dim)

        # Node model
        self.node_net = nn.ModuleDict()
        self.node_net["node_fc0"] = nn.Linear(self.node_dim, self.hidden_dim)
        self.node_net["edge_fc0"] = nn.Linear(self.edge_dim, self.hidden_dim, bias=False)
        self.node_net["fc1"] = nn.Linear(
            self.hidden_dim, self.node_dim + self.graph_dim + use_attention,
        )
        if self.drop_rate > 0:
            self.node_net["dropout"] = nn.Dropout(self.drop_rate)
        if self.normalize:
            self.node_net["node_norm"] = nn.LayerNorm(self.node_dim)
            self.node_net["edge_norm"] = nn.LayerNorm(self.edge_dim)
            self.node_net["hidden_norm"] = nn.LayerNorm(self.hidden_dim)

    def edge_model(
        self,
        node_feats: Tensor,
        edge_feats: Tensor,
        edge_index: LongTensor,
        graph_feats: Tensor,
        graph_index: LongTensor,
    ) -> Tuple[Tensor, Tensor]:
        row, col = edge_index
        row_graph_index, col_graph_index = graph_index[row], graph_index[col]
        assert torch.equal(row_graph_index, col_graph_index)
        (node_feats_row, node_feats_col, edge_feats_, graph_feats_) = (
            node_feats, node_feats, edge_feats, graph_feats
        )
        if self.normalize:
            node_feats_row = self.edge_net.node_row_norm(node_feats_row)
            node_feats_col = self.edge_net.node_col_norm(node_feats_col)
            edge_feats_ = self.edge_net.edge_norm(edge_feats_)
            graph_feats_ = self.edge_net.graph_norm(graph_feats_)
        node_feats_row = self.edge_net.node_row_fc0(node_feats_row.relu())
        node_feats_col = self.edge_net.node_col_fc0(node_feats_col.relu())
        edge_feats_ = self.edge_net.edge_fc0(edge_feats_.relu())
        graph_feats_ = self.edge_net.graph_fc0(graph_feats_.relu())
        carry = (
            node_feats_row[row] + node_feats_col[col] + edge_feats_ + graph_feats_[row_graph_index]
        )
        if self.drop_rate > 0:
            carry = self.edge_net.dropout(carry)
        if self.normalize:
            carry = self.edge_net.hidden_norm(carry)
        carry = self.edge_net.fc1(carry.relu())
        if self.aggregate == "attn":
            edge_feats_, edge_attn_logit = torch.split(carry, [self.edge_dim, 1], dim=1)
            edge_attn = edge_attn_logit.sigmoid()
        else:
            edge_feats_, edge_attn = carry, None
        if self.residual:
            edge_feats_ = edge_feats + edge_feats_
        return edge_feats_, edge_attn

    def node_model(
        self,
        node_feats: Tensor,
        edge_feats: Tensor,
        edge_index: LongTensor,
        graph_feats: Tensor,
        graph_index: LongTensor,
        edge_attn: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        row, col = edge_index
        if self.aggregate == "attn":
            assert edge_attn is not None
            edge_messages = scatter_sum(edge_feats * edge_attn, row, 0)
        else:
            assert edge_attn is None
            if self.aggregate == "max":
                edge_messages = scatter_max(edge_feats, row, 0)[0]
            elif self.aggregate == "mean":
                edge_messages = scatter_mean(edge_feats, row, 0)
            elif self.aggregate == "sum":
                edge_messages = scatter_sum(edge_feats, row, 0)
            else:
                raise ValueError("aggregate must be set to 'attn', 'max', 'mean', or 'sum'.")
        node_feats_ = node_feats
        if self.normalize:
            node_feats_ = self.node_net.node_norm(node_feats_)
            edge_messages = self.node_net.edge_norm(edge_messages)
        node_feats_ = self.node_net.node_fc0(node_feats_.relu())
        edge_messages = self.node_net.edge_fc0(edge_messages.relu())
        carry = node_feats_ + edge_messages
        if self.drop_rate > 0:
            carry = self.node_net.dropout(carry)
        if self.normalize:
            carry = self.node_net.hidden_norm(carry)
        carry = self.node_net.fc1(carry.relu())
        if self.aggregate == "attn":
            node_feats_, carry, node_attn_logit = (
                torch.split(carry, [self.node_dim, self.graph_dim, 1], dim=1)
            )
            node_attn = node_attn_logit.sigmoid()
            graph_feats_ = scatter_sum(carry * node_attn, graph_index, dim=0)
        else:
            node_feats_, carry = torch.split(carry, [self.node_dim, self.graph_dim], dim=1)
            if self.aggregate == "max":
                graph_feats_ = scatter_max(carry, graph_index, dim=0)[0]
            elif self.aggregate == "mean":
                graph_feats_ = scatter_mean(carry, graph_index, dim=0)
            elif self.aggregate == "sum":
                graph_feats_ = scatter_sum(carry, graph_index, dim=0)
            else:
                raise Exception("aggregate must be set to 'attn', 'max', 'mean', or 'sum'.")
        if self.residual:
            node_feats_ = node_feats + node_feats_
            graph_feats_ = graph_feats + graph_feats_
        return node_feats_, graph_feats_

    def forward(
        self,
        node_feats: Tensor,
        edge_feats: Tensor,
        edge_index: LongTensor,
        graph_feats: Tensor,
        graph_index: LongTensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        edge_feats, edge_attn = self.edge_model(
            node_feats, edge_feats, edge_index, graph_feats, graph_index,
        )
        node_feats, graph_feats = self.node_model(
            node_feats, edge_feats, edge_index, graph_feats, graph_index, edge_attn,
        )
        return node_feats, edge_feats, graph_feats


class EdgeMessagePassingNetwork(nn.Module):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        graph_dim: int,
        hidden_dim: int,
        residual: bool = True,
        normalize: bool = True,
        aggregate: str = "sum",
        num_layers: int = 4,
    ):
        super().__init__()
        self.node_dim = int(node_dim)
        self.edge_dim = int(edge_dim)
        self.graph_dim = int(graph_dim)
        self.hidden_dim = int(hidden_dim)
        self.residual = bool(residual)
        self.normalize = bool(normalize)
        self.aggregate = str(aggregate)
        self.num_layers = int(num_layers)
        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            layer = EdgeMessagePassingLayer(
                node_dim=self.node_dim,
                edge_dim=self.edge_dim,
                graph_dim=self.graph_dim,
                hidden_dim=self.hidden_dim,
                residual=self.residual,
                normalize=self.normalize,
                aggregate=self.aggregate,
            )
            self.layers.append(layer)

    def forward(
        self,
        node_feats: Tensor,
        edge_feats: Tensor,
        edge_index: LongTensor,
        graph_feats: Tensor,
        graph_index: LongTensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        for layer in self.layers:
            node_feats, edge_feats, graph_feats = layer(
                node_feats, edge_feats, edge_index, graph_feats, graph_index,
            )
        return node_feats, edge_feats, graph_feats
