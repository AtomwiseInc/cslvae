from torch import LongTensor, nn, Tensor


class ModularizedScatter(nn.Module):
    def __init__(self, scatter_op):
        super().__init__()
        self.scatter_op = scatter_op

    def forward(self, x: Tensor, index: LongTensor) -> Tensor:
        assert index.ndim in [1, 2]
        if index.ndim == 2:
            x = x[index[0]]
            index = index[1]
        return self.scatter_op(x, index, dim=0)
