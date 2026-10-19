"""Transparent native Linear adapters; NF4 requires CUDA bitsandbytes."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class LowRank(nn.Module):
    def __init__(self, base, rank=8, alpha=16., method='lora'):
        super().__init__()
        if rank < 1 or method not in ('lora', 'rslora', 'dora', 'qlora'):
            raise ValueError('Invalid adapter settings')
        self.base, self.method = base, method
        self.scale = alpha / (math.sqrt(rank) if method == 'rslora' else rank)
        self.a = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device))
        self.b = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))
        base.requires_grad_(False)
        if method == 'dora':
            self.magnitude = nn.Parameter(base.weight.detach().norm(dim=1))

    def forward(self, x):
        if self.method == 'dora':
            direction = self.base.weight + self.scale * (self.b @ self.a)
            # DoRA uses a detached direction norm (see official NVlabs implementation).
            norm = direction.norm(dim=1).detach().clamp_min(1e-8)
            return F.linear(x, direction * (self.magnitude / norm)[:, None], self.base.bias)
        return self.base(x) + self.scale * F.linear(F.linear(x, self.a), self.b)


def attach(model, method='lora', rank=8, alpha=16.):
    if method == 'full':
        return model
    model.requires_grad_(False)
    for block in model.blocks:
        for parent, name in ((block.att, 'qkv'), (block.att, 'out'),
                             (block.ff, '0'), (block.ff, '2')):
            base = getattr(parent, name)
            if method == 'qlora':
                import bitsandbytes as bnb
                if not torch.cuda.is_available():
                    raise RuntimeError('NF4 experiment requires CUDA')
                quant = bnb.nn.Linear4bit(base.in_features, base.out_features,
                                         bias=base.bias is not None,
                                         compute_dtype=torch.float32,
                                         compress_statistics=True, quant_type='nf4')
                quant.load_state_dict(base.state_dict())
                base = quant.to('cuda')
            setattr(parent, name, LowRank(base, rank, alpha, method))
    return model


def trainable_state(model):
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}


def load_trainable(model, state):
    params = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in state.items():
            params[name].copy_(value)
