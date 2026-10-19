"""E independent small decoder-only Transformers, trained side by side in one process.

Why an "ensemble"?  Learning time is heavy-tailed, so one seed per condition tells
us almost nothing.  Every parameter tensor here has a leading member axis E:
member e has its own initialization (seed), its own training examples, its own
gradient clipping and its own AdamW moments.  Members never interact; stacking
them only lets one GPU kernel run E models at once.  (test_synth.py checks that
member e gives exactly the same logits as a model built alone with that seed.)

Shapes used in comments:
    E members, B examples per member, T tokens, d width, H heads, hd = d // H.

One layer (pre-LayerNorm, as in GPT-2):
    x = x + Attention(LN1(x))
    x = x + MLP(LN2(x))
"""
from dataclasses import dataclass, field
import math

import torch
import torch.nn.functional as F


@dataclass
class ModelCfg:
    vocab: int = 28
    d: int = 64
    heads: int = 4
    layers: int = 2
    max_len: int = 1100
    pe: str = "rope"            # rope | learned (absolute position table, used only by the v1 configuration)
    slopes: list = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])  # ALiBi-style distance penalty per head
    rope_theta: float = 10000.0
    # Diagnostic intervention (v3, experiment X1).  If k > 0, head 0 of layer 0 is forced to
    # attend ONLY to the token k positions back (t-k).  Its value/output weights are still
    # learned.  k=1 hands the model a "previous-token head"; k=2 is the control.
    fixed_head_offset: int = 0


# ----------------------------------------------------------------------------- init
def _init_member(cfg, seed):
    """Parameters of ONE model.  Same seed -> same model (used for paired comparisons)."""
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g) * 0.02
    d, V = cfg.d, cfg.vocab
    p = {"emb": r(V, d)}
    if cfg.pe == "learned":
        p["pos"] = r(cfg.max_len, d)
    for i in range(cfg.layers):
        p[f"l{i}.ln1s"], p[f"l{i}.ln1b"] = torch.ones(d), torch.zeros(d)
        p[f"l{i}.wqkv"], p[f"l{i}.bqkv"] = r(d, 3 * d), torch.zeros(3 * d)   # q, k, v projections side by side
        p[f"l{i}.wo"], p[f"l{i}.bo"] = r(d, d), torch.zeros(d)
        p[f"l{i}.ln2s"], p[f"l{i}.ln2b"] = torch.ones(d), torch.zeros(d)
        p[f"l{i}.w1"], p[f"l{i}.b1"] = r(d, 4 * d), torch.zeros(4 * d)
        p[f"l{i}.w2"], p[f"l{i}.b2"] = r(4 * d, d), torch.zeros(d)
    p["lnfs"], p["lnfb"] = torch.ones(d), torch.zeros(d)
    p["head"] = r(d, V)                                                        # untied unembedding
    return p


def init_params(cfg, seeds):
    """{name: tensor [E, ...]} for the given member seeds (also used to re-initialize layers)."""
    members = [_init_member(cfg, s) for s in seeds]
    return {k: torch.stack([m[k] for m in members]) for k in members[0]}


# ----------------------------------------------------------------------------- model
class Ensemble(torch.nn.Module):
    def __init__(self, cfg: ModelCfg, seeds, device):
        super().__init__()
        self.cfg, self.seeds, self.E = cfg, list(seeds), len(seeds)
        # ParameterDict keys cannot contain ".", so "l0.wqkv" is stored as "l0_wqkv".
        self.params = torch.nn.ParameterDict(
            {k.replace(".", "_"): torch.nn.Parameter(v.to(device)) for k, v in init_params(cfg, self.seeds).items()})
        # RoPE tables: pair (2j, 2j+1) of every head is rotated by angle  position * theta^(-2j/hd).
        hd = cfg.d // cfg.heads
        inv_freq = 1.0 / cfg.rope_theta ** (torch.arange(0, hd, 2).float() / hd)
        angle = torch.arange(cfg.max_len).float()[:, None] * inv_freq[None]         # [max_len, hd/2]
        self.register_buffer("cos", angle.cos().to(device), persistent=False)
        self.register_buffer("sin", angle.sin().to(device), persistent=False)
        self.register_buffer("slopes", torch.tensor(cfg.slopes, dtype=torch.float32, device=device),
                             persistent=False)

    def P(self, name):
        return self.params[name.replace(".", "_")]

    # ---- building blocks (every op acts on each member separately) ----
    @staticmethod
    def _layernorm(x, scale, bias):                 # x [E,B,T,d]; normalize each token's d features
        x = x - x.mean(-1, keepdim=True)
        x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-5)
        return x * scale[:, None, None] + bias[:, None, None]

    @staticmethod
    def _linear(x, w, b=None):                      # x [E,B,T,i] @ w [E,i,o] -> [E,B,T,o]
        E, B, T, i = x.shape
        y = torch.bmm(x.reshape(E, B * T, i), w).reshape(E, B, T, -1)
        return y if b is None else y + b[:, None, None]

    def _rope(self, z):                             # z [E,B,H,T,hd]: rotate each (even, odd) pair
        T = z.shape[-2]
        c, s = self.cos[:T], self.sin[:T]
        z1, z2 = z[..., 0::2], z[..., 1::2]
        return torch.stack([z1 * c - z2 * s, z1 * s + z2 * c], -1).flatten(-2)

    def _score_bias(self, T, window, device):
        """Additive attention bias per layer, each [H,T,T] (0 = allowed, -inf = forbidden),
        or None when plain causal attention suffices (then the fused SDPA kernel is used)."""
        cfg = self.cfg
        special = window is not None or bool((self.slopes != 0).any()) or cfg.fixed_head_offset > 0
        if not special:
            return None
        pos = torch.arange(T, device=device)
        dist = pos[:, None] - pos[None, :]          # dist[t, s] = t - s  (how far back key s is from query t)
        allowed = dist >= 0                         # causal: never look at the future
        if window is not None:
            allowed = allowed & (dist < window)     # local attention: only the last `window` tokens
        bias = -self.slopes[:, None, None] * dist.clamp(min=0)[None].float()       # [H,T,T]
        bias = bias.masked_fill(~allowed[None], float("-inf"))
        per_layer = [bias] * cfg.layers
        k = cfg.fixed_head_offset
        if k > 0:                                   # layer 0, head 0: only key t-k (t<k: itself)
            only = (dist == k) | ((pos[:, None] < k) & (dist == 0))
            b0 = bias.clone()
            b0[0] = torch.zeros(T, T, device=device).masked_fill(~only, float("-inf"))
            per_layer[0] = b0
        return per_layer

    def _attention(self, i, h, bias, block, keep_weights):
        """Multi-head causal self-attention of layer i.  h [E,B,T,d] -> ([E,B,T,d], weights or None)."""
        cfg, E = self.cfg, self.E
        _, B, T, _ = h.shape
        H, hd = cfg.heads, cfg.d // cfg.heads
        q, k, v = self._linear(h, self.P(f"l{i}.wqkv"), self.P(f"l{i}.bqkv")).chunk(3, -1)
        q, k, v = [z.reshape(E, B, T, H, hd).transpose(2, 3) for z in (q, k, v)]   # [E,B,H,T,hd]
        if cfg.pe == "rope":
            q, k = self._rope(q), self._rope(k)
        weights = None
        if bias is None and block is None and not keep_weights:
            # fast path: fused kernel, mathematically the same as the manual path below
            o = F.scaled_dot_product_attention(q.reshape(E * B, H, T, hd), k.reshape(E * B, H, T, hd),
                                               v.reshape(E * B, H, T, hd), is_causal=True).reshape(E, B, H, T, hd)
        else:
            if bias is None:
                bias = torch.zeros(T, T, device=h.device).masked_fill(
                    torch.ones(T, T, dtype=torch.bool, device=h.device).triu(1), float("-inf"))
            scores = (q @ k.transpose(-1, -2)) / math.sqrt(hd) + bias                # [E,B,H,T,T]
            if block is not None:                    # diagnostic ablation: forbid extra (query, key) pairs
                scores = scores.masked_fill(block[None, :, None], float("-inf"))
            weights = torch.softmax(scores, -1)      # each row sums to 1 over the allowed keys
            o = weights @ v
        o = o.transpose(2, 3).reshape(E, B, T, cfg.d)                                   # concatenate heads
        return self._linear(o, self.P(f"l{i}.wo"), self.P(f"l{i}.bo")), (weights if keep_weights else None)

    def _mlp(self, i, h):
        h = F.gelu(self._linear(h, self.P(f"l{i}.w1"), self.P(f"l{i}.b1")), approximate="tanh")
        return self._linear(h, self.P(f"l{i}.w2"), self.P(f"l{i}.b2"))

    def forward(self, ids, window=None, attn=False, hidden=False, block=None):
        """ids [E,B,T] (member e reads ids[e]) -> logits [E,B,T,V].
        attn=True   also return per-layer attention weights, each [E,B,H,T,T]
        hidden=True also return the residual stream after each layer, each [E,B,T,d]
        block       diagnostics only: list with one entry per layer, None or bool [B,T,T]
                    marking extra forbidden (query position, key position) pairs."""
        cfg, E = self.cfg, self.E
        T = ids.shape[-1]
        member = torch.arange(E, device=ids.device)[:, None, None]
        x = self.P("emb")[member, ids] * math.sqrt(cfg.d)                            # [E,B,T,d]
        if cfg.pe == "learned":
            x = x + self.P("pos")[:, None, :T]
        biases = self._score_bias(T, window, ids.device)
        maps, states = [], []
        for i in range(cfg.layers):
            a, w = self._attention(i, self._layernorm(x, self.P(f"l{i}.ln1s"), self.P(f"l{i}.ln1b")),
                                   None if biases is None else biases[i],
                                   None if block is None else block[i], keep_weights=attn)
            x = x + a
            x = x + self._mlp(i, self._layernorm(x, self.P(f"l{i}.ln2s"), self.P(f"l{i}.ln2b")))
            maps.append(w)
            states.append(x)
        logits = self._linear(self._layernorm(x, self.P("lnfs"), self.P("lnfb")), self.P("head"))
        out = [logits] + ([maps] if attn else []) + ([states] if hidden else [])
        return out[0] if len(out) == 1 else tuple(out)


# ----------------------------------------------------------------------------- optimizer
class EnsembleAdamW:
    """Plain AdamW (decoupled weight decay), applied elementwise, so members stay independent.
    Gradient clipping is by each member's own global norm."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, wd=0.01):
        self.params = list(params)
        self.lr, self.betas, self.eps, self.wd = lr, betas, eps, wd
        self.t = 0
        self.m = [torch.zeros_like(p) for p in self.params]     # first moment
        self.v = [torch.zeros_like(p) for p in self.params]     # second moment

    @torch.no_grad()
    def clip(self, max_norm=1.0):
        grads = [p.grad for p in self.params]
        norm = torch.stack([g.flatten(1).square().sum(1) for g in grads]).sum(0).sqrt()   # [E]
        scale = (max_norm / (norm + 1e-6)).clamp(max=1.0)
        torch._foreach_mul_(grads, [scale.view(-1, *([1] * (g.ndim - 1))) for g in grads])
        return norm

    @torch.no_grad()
    def step(self):
        """m = b1 m + (1-b1) g;  v = b2 v + (1-b2) g^2;
        p = p (1 - lr wd) - lr * m_hat / (sqrt(v_hat) + eps)   (written with fused multi-tensor ops)."""
        self.t += 1
        b1, b2 = self.betas
        grads = [p.grad for p in self.params]
        torch._foreach_mul_(self.m, b1)
        torch._foreach_add_(self.m, grads, alpha=1 - b1)
        torch._foreach_mul_(self.v, b2)
        torch._foreach_addcmul_(self.v, grads, grads, value=1 - b2)
        torch._foreach_mul_(self.params, 1 - self.lr * self.wd)
        denom = torch._foreach_sqrt(torch._foreach_div(self.v, 1 - b2 ** self.t))
        torch._foreach_add_(denom, self.eps)
        torch._foreach_addcdiv_(self.params, self.m, denom, value=-self.lr / (1 - b1 ** self.t))

    def state(self):
        return {"m": self.m, "v": self.v, "t": self.t}

    def load(self, s):
        self.m, self.v, self.t = s["m"], s["v"], s["t"]
