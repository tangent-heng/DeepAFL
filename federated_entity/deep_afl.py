from typing import Optional, Dict, Literal, Callable, Type
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
from torch import Tensor


class RidgeModule(nn.Module):
    def __init__(self, l2_reg: float = 1e-3):
        super(RidgeModule, self).__init__()
        self.l2_reg = l2_reg
        self.W =None

    def fit(self, corr_X: Tensor, corr_Y: Tensor):
        A = corr_X + self.l2_reg * torch.eye(corr_X.size(0), dtype=corr_X.dtype, device=corr_X.device)
        B = corr_Y
        self.W = torch.inverse(A) @ B
        return self

    def forward(self, X: Tensor) -> Tensor:
        return X @ self.W


def create_layer(
        in_dim: int, 
        out_dim: int,
        activation: nn.Module = nn.Tanh(),
        ):
    layer = nn.Sequential(
        nn.Linear(in_dim, out_dim),
        activation,
    )

    return layer


class RandFeat0(nn.Module):
    def __init__(self,
                 in_dim: int,
                 hidden_dim: int = 128,
                 activation: nn.Module = nn.Tanh(),
                 ):
        super(RandFeat0, self).__init__()
        self.rf_0 = create_layer(in_dim, hidden_dim, activation)

    def forward(self, X: Tensor):
        return self.rf_0(X)




class RandFeatLayer(nn.Module):
    def __init__(self,
                 hidden_dim: int,
                 randfeat_xt_dim: int,
                 activation: nn.Module = nn.Tanh(),
                 ):
        self.hidden_dim = hidden_dim
        self.randfeat_xt_dim = randfeat_xt_dim
        super(RandFeatLayer, self).__init__()

        self.Ft = create_layer(hidden_dim, randfeat_xt_dim, activation)
    

    def forward(self, Xt: Tensor) -> Tensor:
        return self.Ft(Xt)



class SandwichedLayer(nn.Module):
    def __init__(self,
                 hidden_dim: int = 128,
                 l2_sand: float = 0.01,
                 ):
        self.hidden_dim = hidden_dim
        self.l2_sand = l2_sand
        super(SandwichedLayer, self).__init__()

    def fit(self, A, B, W):
        self.Delta = self.sandwiched_LS(A, B, W, self.l2_sand)

    def sandwiched_LS(self, FF, WRF, W, l2_reg: float = 0.01):
        SW, U = torch.linalg.eigh(W @ W.T)
        SX, V = torch.linalg.eigh(FF)
        Delta = torch.linalg.multi_dot((U.T, WRF, V)) / (l2_reg + SW[:, None] * SX[None, :])
        Delta = U @ Delta @ V.T
        return Delta.T


    def forward(self, F: Tensor) -> Tensor:
        return F @ self.Delta

class DeepAFL(nn.Module):
    def __init__(
            self,
            in_dim: int,
            out_dim: int,
            hidden_dim: int = 128,
            n_layers: int = 5,
            randfeat_xt_dim: int = 512,
            l2_reg: Optional[float] = 0.0001,
            l2_sand: Optional[float] = 0.0001,
            activation = "tanh",
    ):
        super(DeepAFL, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.randfeat_xt_dim = randfeat_xt_dim
        self.l2_reg = l2_reg
        self.l2_sand = l2_sand

        if not l2_reg:
            l2_reg = l2_sand
        elif not l2_sand:
            l2_sand = l2_reg

        if isinstance(activation, str):
            if activation.lower() == "tanh":
                activation = nn.Tanh()
            elif activation.lower() == "relu":
                activation = nn.ReLU()
            elif activation.lower() == "gelu":
                activation = nn.GELU()
            elif activation.lower() == "sigmoid":
                activation = nn.Sigmoid()
            elif activation.lower() == "identity":
                activation = nn.Identity()
            elif activation.lower() == "hardwish":
                activation = nn.Hardswish()
            elif activation.lower() == "leakyrelu":
                activation = nn.LeakyReLU()
            elif activation.lower() == "softshrink":
                activation = nn.Softshrink()
            else:
                raise ValueError(f"Unknown activation {activation}")

        self.rf_0 = RandFeat0(in_dim, hidden_dim, nn.Tanh())

        top_classifiers = [RidgeModule(l2_reg) for _ in range(n_layers + 1)]
        self.top_classifiers = nn.ModuleList(top_classifiers)

        rf_layers = [
            RandFeatLayer(hidden_dim, randfeat_xt_dim, activation)
            for _ in range(n_layers)
        ]
        self.rf_layers = nn.ModuleList(rf_layers)

        sandwiched_layers = [
            SandwichedLayer(hidden_dim, l2_sand)
            for _ in range(n_layers)
        ]
        self.sandwiched_layers = nn.ModuleList(sandwiched_layers)

    def update_top_classifiers(self, corr_X, corr_Y, n: int):
        self.top_classifiers[n].fit(corr_X, corr_Y)

    def update_sandwiched_layers(self, A, B, W, n: int):
        self.sandwiched_layers[n].fit(A, B, W)

    def forward(self, X: Tensor) -> Dict[int, Tensor]:
        X0 = X
        X = self.rf_0(X0)
        result: Dict[int, Tensor] = {}
        i = 0
        result[i] = X
        for randfeat_layer, sandwiched_layer in zip(self.rf_layers, self.sandwiched_layers):
            F = randfeat_layer(X)
            sandwiched_feature = sandwiched_layer(F)
            X = X + sandwiched_feature
            i += 1
            result[i] = X

        for k in list(result.keys()):
            result[k] = self.top_classifiers[k](result[k])

        return result


