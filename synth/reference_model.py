"""The same Transformer as ensemble.py, written as ONE ordinary model for reading.

Experiments use ensemble.py (E seeds at once).  This file exists so the math can be read
without the extra member axis; test_synth.py checks that loading member e of an
ensemble checkpoint into this class gives the same logits.

Shapes: B batch, T tokens, d width, H heads, hd = d // H, V vocabulary.
"""
import math

import torch
import torch.nn.functional as F


class ReferenceTransformer(torch.nn.Module):
    def __init__(self, params, cfg):
        """params: {name: tensor} for ONE model (names as in ensemble._init_member); cfg: ModelCfg."""
        super().__init__()
        self.cfg = cfg
        self.p = {k: v for k, v in params.items()}
        hd = cfg.d // cfg.heads
        inv_freq = 1.0 / cfg.rope_theta ** (torch.arange(0, hd, 2).float() / hd)
        angle = torch.arange(cfg.max_len).float()[:, None] * inv_freq[None]       # [max_len, hd/2]
        self.cos, self.sin = angle.cos(), angle.sin()

    @classmethod
    def from_checkpoint(cls, ckpt, member, device="cpu"):
        from ensemble import ModelCfg
        st = torch.load(ckpt, map_location=device, weights_only=False)
        cfg = ModelCfg(**{**st["meta"]["model"], "max_len": 1100})
        params = {k.replace("_", ".", 1): v[member].float() for k, v in st["params"].items()}
        return cls(params, cfg)

    def layernorm(self, x, name):
        mean = x.mean(-1, keepdim=True)
        var = (x - mean).square().mean(-1, keepdim=True)
        return (x - mean) / torch.sqrt(var + 1e-5) * self.p[name + "s"] + self.p[name + "b"]

    def rope(self, z):                         # z [B,H,T,hd]: rotate pairs (0,1), (2,3), ... by pos * freq
        T = z.shape[-2]
        cos, sin = self.cos[:T].to(z), self.sin[:T].to(z)
        even, odd = z[..., 0::2], z[..., 1::2]
        return torch.stack([even * cos - odd * sin, even * sin + odd * cos], -1).flatten(-2)

    def attention(self, i, h):
        B, T, d = h.shape
        H, hd = self.cfg.heads, d // self.cfg.heads
        qkv = h @ self.p[f"l{i}.wqkv"] + self.p[f"l{i}.bqkv"]                    # [B,T,3d]
        q, k, v = qkv.split(d, dim=-1)
        q, k, v = [z.view(B, T, H, hd).transpose(1, 2) for z in (q, k, v)]       # [B,H,T,hd]
        if self.cfg.pe == "rope":
            q, k = self.rope(q), self.rope(k)
        scores = q @ k.transpose(-1, -2) / math.sqrt(hd)                         # [B,H,T,T] query t, key s
        t = torch.arange(T)
        dist = t[:, None] - t[None, :]                                           # t - s
        scores = scores - torch.tensor(self.cfg.slopes).to(h)[:, None, None] * dist.clamp(min=0)
        allowed = dist >= 0                                                      # causal mask
        if i == 0 and self.cfg.fixed_head_offset:                                # X1 diagnostic head
            k_off = self.cfg.fixed_head_offset
            only = (dist == k_off) | ((t[:, None] < k_off) & (dist == 0))
            scores[:, 0] = torch.where(only, 0.0, float("-inf"))
        scores = scores.masked_fill(~allowed, float("-inf"))
        weights = scores.softmax(-1)                                             # rows sum to 1
        out = (weights @ v).transpose(1, 2).reshape(B, T, d)                     # concat heads
        return out @ self.p[f"l{i}.wo"] + self.p[f"l{i}.bo"], weights

    def mlp(self, i, h):
        h = F.gelu(h @ self.p[f"l{i}.w1"] + self.p[f"l{i}.b1"], approximate="tanh")
        return h @ self.p[f"l{i}.w2"] + self.p[f"l{i}.b2"]

    def forward(self, ids):                     # ids [B,T] -> logits [B,T,V], attention maps per layer
        x = self.p["emb"][ids] * math.sqrt(self.cfg.d)
        if self.cfg.pe == "learned":
            x = x + self.p["pos"][: ids.shape[1]]
        maps = []
        for i in range(self.cfg.layers):
            a, w = self.attention(i, self.layernorm(x, f"l{i}.ln1"))
            x = x + a                                          # residual 1
            x = x + self.mlp(i, self.layernorm(x, f"l{i}.ln2"))  # residual 2
            maps.append(w)
        return self.layernorm(x, "lnf") @ self.p["head"], maps
