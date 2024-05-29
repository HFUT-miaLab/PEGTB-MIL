import torch.nn as nn
from models.nystrom_attention import NystromAttention


class TransLayer(nn.Module):

    def __init__(self, norm_layer=nn.LayerNorm, dim=512, return_attn=False):
        super().__init__()
        self.return_attn = return_attn
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim=dim,
            dim_head=dim // 8,
            heads=8,
            num_landmarks=dim // 2,  # number of landmarks
            pinv_iterations=6,
            # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual=True,
            # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=0.1
        )

    def forward(self, x):
        if self.return_attn:
            r, att = self.attn(self.norm(x), return_attn=True)
            x = x + r
            return att, x
        else:
            r = self.attn(self.norm(x), return_attn=False)
            x = x + r
            return None, x
