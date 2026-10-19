from dataclasses import dataclass
import torch
from torch import nn


@dataclass
class Config:
    vocab_size: int = 50257
    context_length: int = 256
    emb_dim: int = 128
    n_heads: int = 4
    n_layers: int = 4
    dropout: float = 0.1
    qkv_bias: bool = False
    tie_embeddings: bool = False


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        if cfg.emb_dim % cfg.n_heads:
            raise ValueError("emb_dim must be divisible by n_heads")
        self.heads = cfg.n_heads
        self.qkv = nn.Linear(cfg.emb_dim, 3 * cfg.emb_dim, bias=cfg.qkv_bias)
        self.out = nn.Linear(cfg.emb_dim, cfg.emb_dim)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x, past=None):
        b, t, d = x.shape
        q, k, v = [z.reshape(b, t, self.heads, d // self.heads).transpose(1, 2)
                   for z in self.qkv(x).chunk(3, dim=-1)]
        offset = 0 if past is None else past[0].size(2)
        if past is not None:
            k, v = torch.cat((past[0], k), 2), torch.cat((past[1], v), 2)
        scores = (q @ k.transpose(-2, -1)) / (d // self.heads) ** 0.5
        mask = torch.arange(k.size(2), device=x.device)[None, :] > (
            offset + torch.arange(t, device=x.device)[:, None])
        weights = self.dropout(scores.masked_fill(mask, float("-inf")).softmax(-1))
        out = (weights @ v).transpose(1, 2).contiguous().reshape(b, t, d)
        return self.out(out), (k, v)


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.emb_dim)
        self.norm2 = nn.LayerNorm(cfg.emb_dim)
        self.att = Attention(cfg)
        self.ff = nn.Sequential(nn.Linear(cfg.emb_dim, 4 * cfg.emb_dim),
                                nn.GELU(approximate="tanh"),
                                nn.Linear(4 * cfg.emb_dim, cfg.emb_dim))
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, past=None):
        a, cache = self.att(self.norm1(x), past)
        x = x + self.drop(a)
        return x + self.drop(self.ff(self.norm2(x))), cache


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.emb_dim)
        self.pos_emb = nn.Embedding(cfg.context_length, cfg.emb_dim)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm = nn.LayerNorm(cfg.emb_dim)
        self.head = nn.Linear(cfg.emb_dim, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.head.weight = self.tok_emb.weight

    def hidden(self, ids, past=None):
        if past is not None and len(past) != len(self.blocks):
            raise ValueError("One cache entry is required per layer")
        offset = 0 if past is None else past[0][0].size(2)
        if ids.size(1) == 0 or offset + ids.size(1) > self.cfg.context_length:
            raise ValueError("Empty input or context length exceeded")
        x = self.drop(self.tok_emb(ids) + self.pos_emb(
            torch.arange(offset, offset + ids.size(1), device=ids.device)))
        caches = []
        for i, block in enumerate(self.blocks):
            x, cache = block(x, None if past is None else past[i])
            caches.append(cache)
        return self.norm(x), caches

    def forward(self, ids, past=None, use_cache=False):
        x, cache = self.hidden(ids, past)
        logits = self.head(x)
        return (logits, cache) if use_cache else logits


class Classifier(nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(backbone.cfg.emb_dim, num_classes)

    def forward(self, ids, lengths):
        x, _ = self.backbone.hidden(ids)
        return self.head(x[torch.arange(ids.size(0), device=ids.device), lengths - 1])

    def configure(self, mode):
        if mode not in ("head", "last", "full"):
            raise ValueError(mode)
        for p in self.backbone.parameters():
            p.requires_grad = mode == "full"
        # Language-model head is unused by classification.
        if self.backbone.head.weight is not self.backbone.tok_emb.weight:
            for p in self.backbone.head.parameters():
                p.requires_grad = False
        if mode == "last":
            for module in (self.backbone.blocks[-1], self.backbone.norm):
                for p in module.parameters():
                    p.requires_grad = True
