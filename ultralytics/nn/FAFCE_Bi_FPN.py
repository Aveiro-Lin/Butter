import torch
import torch.nn as nn

__all__ = ["Bi_FPN"]


class swish(nn.Module):


    def forward(self, x):
        return x * torch.sigmoid(x)


class Bi_FPN(nn.Module):


    def __init__(self, length):
        super().__init__()
        # keep attribute names identical for state_dict compatibility
        self.weight = nn.Parameter(torch.ones(length, dtype=torch.float32), requires_grad=True)
        self.swish = swish()
        self.epsilon = 0.0001

    def _normalized_weights(self):
        # identical math to original: denom uses sum(swish(weight)) + epsilon
        denom = torch.sum(self.swish(self.weight), dim=0) + self.epsilon
        return self.weight / denom

    def forward(self, x):
        weights = self._normalized_weights()
        weighted_feature_maps = [feat * weights[i] for i, feat in enumerate(x)]
        return torch.stack(weighted_feature_maps, dim=0).sum(dim=0)