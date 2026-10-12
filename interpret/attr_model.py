"""ConvTowerDomain_v6 rewritten so that every non-linearity is an nn.Module tangermeme's DeepLIFT rules can hook.

nn.TransformerEncoderLayer runs attention and its FFN activation as functional calls, so it is replaced by
AttributableEncoderLayer, which computes the same function from the same weights.
"""

import math
import os
import sys

import torch
from einops.layers.torch import Rearrange
from torch import nn
from tangermeme.deep_lift_shap import BilinearOp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import ConvTowerDomain_v6  # noqa: E402

# models.py sets 'medium', and cuDNN convolutions default to TF32; either breaks the 1e-3 convergence checks on GPU
torch.set_float32_matmul_precision("highest")
torch.backends.cudnn.allow_tf32 = False

DEFAULT_CKPT = "checkpoints/FOXA1_A549_rep1_center_win240_wd1e-4/best-epoch=40-val_loss=0.344953.ckpt"


class AttributableEncoderLayer(nn.Module):
    """Post-norm nn.TransformerEncoderLayer (batch_first=False) in eval mode.

    DeepLIFT stacks each example with its reference along dim 0, and the Softmax, LayerNorm and BilinearOp
    rules split along that dim, so the computation runs batch-first internally.
    """

    def __init__(self, layer: nn.TransformerEncoderLayer):
        super().__init__()
        if layer.norm_first or layer.self_attn.batch_first:
            raise ValueError("only post-norm, sequence-first layers are supported")
        attn = layer.self_attn
        self.d_model = attn.embed_dim
        self.n_heads = attn.num_heads
        self.d_head = self.d_model // self.n_heads

        self.q_proj, self.k_proj, self.v_proj = (nn.Linear(self.d_model, self.d_model) for _ in range(3))
        with torch.no_grad():
            for i, proj in enumerate((self.q_proj, self.k_proj, self.v_proj)):
                proj.weight.copy_(attn.in_proj_weight[i * self.d_model:(i + 1) * self.d_model])
                proj.bias.copy_(attn.in_proj_bias[i * self.d_model:(i + 1) * self.d_model])
        self.out_proj = layer.self_attn.out_proj
        self.qk = BilinearOp()
        self.softmax = nn.Softmax(dim=-1)
        self.av = BilinearOp()

        self.norm1 = layer.norm1
        self.linear1 = layer.linear1
        self.activation = nn.ReLU()
        self.linear2 = layer.linear2
        self.norm2 = layer.norm2

        self.to_batch_first = Rearrange("l b c -> b l c")
        self.to_seq_first = Rearrange("b l c -> l b c")

    def _heads(self, x):
        b, l, _ = x.shape
        return x.reshape(b, l, self.n_heads, self.d_head).transpose(1, 2)

    def forward(self, src):
        x = self.to_batch_first(src)
        b, l, _ = x.shape
        q = self._heads(self.q_proj(x)) / math.sqrt(self.d_head)
        k = self._heads(self.k_proj(x))
        v = self._heads(self.v_proj(x))
        attn = self.softmax(self.qk(q, k.transpose(-1, -2)))
        a = self.av(attn, v).transpose(1, 2).reshape(b, l, self.d_model)
        x = self.norm1(x + self.out_proj(a))
        x = self.norm2(x + self.linear2(self.activation(self.linear1(x))))
        return self.to_seq_first(x)


class SeqOnly(nn.Module):
    """forward(seq) -> logit of shape (N, 1); chrom is unused when seqonly=True."""

    def __init__(self, model: ConvTowerDomain_v6):
        super().__init__()
        self.model = model

    def forward(self, seq):
        chrom = torch.zeros(seq.shape[0], 1, seq.shape[2], dtype=seq.dtype, device=seq.device)
        return self.model(seq, chrom)


def load_pretrained(ckpt=DEFAULT_CKPT):
    ckpt = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = ConvTowerDomain_v6(**ckpt["hyper_parameters"])
    model.load_state_dict(ckpt["state_dict"], strict=True)
    if not model.seqonly:
        raise ValueError("expected a seqonly=True checkpoint")
    return model.float().eval()


def build_original_model(ckpt=DEFAULT_CKPT):
    return SeqOnly(load_pretrained(ckpt)).eval()


def build_attr_model(ckpt=DEFAULT_CKPT):
    model = load_pretrained(ckpt)
    model.transformer_encoder = nn.Sequential(*[AttributableEncoderLayer(l) for l in model.transformer_encoder])
    return SeqOnly(model).eval()


@torch.no_grad()
def check_equivalence(original, attr, X, device, batch_size=128, atol=1e-4):
    original.to(device)
    attr.to(device)
    max_diff = 0.0
    for i in range(0, len(X), batch_size):
        x = X[i:i + batch_size].to(device)
        max_diff = max(max_diff, (original(x) - attr(x)).abs().max().item())
    print(f"equivalence check on {len(X)} sequences: max |diff| = {max_diff:.2e}")
    if max_diff >= atol:
        raise RuntimeError(f"attributable model deviates from the original by {max_diff:.2e} (>= {atol})")
    return max_diff


@torch.no_grad()
def predict_logits(model, X, device, batch_size=256):
    model.to(device)
    return torch.cat([model(X[i:i + batch_size].to(device)).cpu() for i in range(0, len(X), batch_size)])[:, 0]
