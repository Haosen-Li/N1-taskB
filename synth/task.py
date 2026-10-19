"""Synthetic associative recall: distance, context length and competitor type are separate knobs.

One example (layout "qk"; 4 records; N noise tokens in total):

    BOS  [a noise]  STORE K V  STORE K V  STORE K V  STORE K V  [b noise]  QUERY K_q   -> answer V_q
    0    1..a       <------------- 12 record tokens ------------>                  ^ final position

    a + b = N            context length  T = 1 + N + 12 + 2 = N + 15   (4 records; n_records=R gives 3R)
    b ~ U[post_lo, post_hi]  (per example)  -> controls the dependency distance
    near:  b in [0, 8]          records sit right before the query, the noise is BEFORE them
    far:   b in [N-8, N]        records sit right after BOS,         the noise is AFTER them

    noise="value"   noise tokens are drawn from the 8 VALUE tokens  (they look like answers: confusable)
    noise="filler"  noise tokens are 8 separate FILLER tokens       (never an answer: inert)
    noise="mix"     n_confusable value tokens + filler for the rest

Concrete example (python show_task.py --N 6 --post_lo 2 --post_hi 2), queried value in [ ]:
    BOS v0 v1 v4 v3 STORE k2 [v0] STORE k3 v4 STORE k4 v2 STORE k6 v1 v5 v3 QUERY k2   -> v0

Pairing: all random numbers are drawn in a fixed order with fixed shapes, whatever the
condition.  So two conditions built from the same generator seed are paired example by
example; e.g. "value" vs "filler" differ ONLY in which token sits at each noise position.
"""
from dataclasses import dataclass
import torch

# ---- vocabulary (28 tokens) ----
BOS, STORE, QUERY, READOUT = 0, 1, 2, 3       # READOUT is only used by the v1 layouts
KEY0, VAL0, FILL0, NV = 4, 12, 20, 8           # keys 4..11, values 12..19, fillers 20..27
VOCAB = 28
N_RECORDS = 4                                  # default number of records (TaskCfg.n_records)


@dataclass
class TaskCfg:
    N: int = 64                 # total number of noise tokens
    post_lo: int = 64           # min noise tokens between the last record and the query
    post_hi: int = 64           # max noise tokens between the last record and the query
    noise: str = "value"        # value | filler | mix
    n_confusable: int = 0       # for noise="mix"
    layout: str = "qk"          # qk: "... QUERY K -> V";  v1direct / v1delayed: the v1 formats
    n_records: int = N_RECORDS  # number of STORE K V records (<= 8, keys are distinct)

    @property
    def rec_len(self):
        return 3 * self.n_records

    @property
    def suffix_len(self):
        return 2 if self.layout == "qk" else 3

    @property
    def T(self):
        return 1 + self.N + self.rec_len + self.suffix_len


def make_batch(n, g, device, c: TaskCfg):
    """n examples.  g: torch.Generator (its state advances).  Returns a dict of tensors:
        x [n,T] input tokens        y [n,T] next-token targets (y[:, -1] = answer)
        answer [n]   keys/values [n,R]   slot [n] which record is queried   (R = n_records, default 4)
        vpos [n,R] positions of the R stored values   src [n] position of the queried value
        qpos         position of the query key        distance [n] = (T-1) - src
        noise_mask / confusable [n,T] which positions are noise / confusable noise
    """
    T, dev, R = c.T, device, c.n_records
    assert 1 <= R <= NV, "keys must be distinct"
    # 1) random draws -- ALWAYS in this order and with these shapes (keeps conditions paired)
    keys = torch.rand(n, NV, generator=g, device=dev).argsort(-1)[:, :R] + KEY0           # R distinct keys
    values = torch.randint(NV, (n, R), generator=g, device=dev) + VAL0                    # values may repeat
    slot = torch.randint(R, (n,), generator=g, device=dev)                                # which key is asked
    post = torch.randint(c.post_lo, c.post_hi + 1, (n,), generator=g, device=dev)        # noise after records
    noise_ids = torch.randint(NV, (n, T), generator=g, device=dev)                       # identity of each noise token
    u = torch.rand(n, T, generator=g, device=dev)                                         # used by noise="mix"

    rows = torch.arange(n, device=dev)
    # 2) where the record block goes: after BOS and (N - post) leading noise tokens
    rec_start = 1 + (c.N - post)                                                          # [n]
    rec_idx = rec_start[:, None] + torch.arange(c.rec_len, device=dev)                   # [n,3R]

    # 3) every position that is not BOS, a record token or the query suffix is noise
    noise_mask = torch.ones(n, T, dtype=torch.bool, device=dev)
    noise_mask[:, 0] = False
    noise_mask[:, T - c.suffix_len:] = False
    noise_mask[rows[:, None], rec_idx] = False
    if c.noise == "value":
        confusable = noise_mask
    elif c.noise == "filler":
        confusable = torch.zeros_like(noise_mask)
    elif c.noise == "mix":              # the n_confusable noise positions with the smallest u
        if c.n_confusable:
            uu = u.masked_fill(~noise_mask, float("inf"))
            kth = uu.topk(c.n_confusable, dim=1, largest=False).values[:, -1:]
            confusable = (uu <= kth) & noise_mask
        else:
            confusable = torch.zeros_like(noise_mask)
    else:
        raise ValueError(c.noise)

    # 4) write the sequence
    x = torch.where(confusable, noise_ids + VAL0, noise_ids + FILL0)
    x[:, 0] = BOS
    records = torch.stack([torch.full_like(keys, STORE), keys, values], -1).flatten(1)   # [n,3R]
    x[rows[:, None], rec_idx] = records
    q = keys[rows, slot]
    if c.layout == "qk":
        x[:, -2], x[:, -1] = QUERY, q
    elif c.layout == "v1direct":
        x[:, -3], x[:, -2], x[:, -1] = QUERY, READOUT, q
    elif c.layout == "v1delayed":
        x[:, -3], x[:, -2], x[:, -1] = QUERY, q, READOUT
    else:
        raise ValueError(c.layout)

    answer = values[rows, slot]
    y = torch.cat([x[:, 1:], answer[:, None]], 1)                     # next-token targets
    vpos = rec_start[:, None] + 2 + 3 * torch.arange(R, device=dev)           # STORE K [V] -> offset 2
    qpos = T - (2 if c.layout == "v1delayed" else 1)
    src = vpos[rows, slot]
    return dict(x=x, y=y, answer=answer, keys=keys, values=values, slot=slot, vpos=vpos, src=src,
                qpos=qpos, noise_mask=noise_mask, confusable=confusable, distance=(T - 1) - src)


# ---- counterfactual inputs used by evaluation (each changes exactly one token) ----
def relevant_edit(b):
    """Change the queried record's value.  A model that really reads it must change its answer."""
    x = b["x"].clone()
    rows = torch.arange(len(x), device=x.device)
    new = (b["answer"] - VAL0 + 1) % NV + VAL0
    x[rows, b["src"]] = new
    return x, new


def irrelevant_edit(b):
    """Change a value that is NOT queried.  The answer should stay the same."""
    x = b["x"].clone()
    rows = torch.arange(len(x), device=x.device)
    other = (b["slot"] + 1) % b["vpos"].shape[1]
    pos = b["vpos"][rows, other]
    x[rows, pos] = (x[rows, pos] - VAL0 + 1) % NV + VAL0
    return x


def query_edit(b):
    """Ask for a different stored key.  A query-blind model cannot follow this."""
    x = b["x"].clone()
    rows = torch.arange(len(x), device=x.device)
    other = (b["slot"] + 1) % b["vpos"].shape[1]
    x[:, b["qpos"]] = b["keys"][rows, other]
    return x, b["values"][rows, other]


def decode(ids):
    """Token ids -> readable string, e.g. 'BOS v3 STORE k2 v5 ... QUERY k6'."""
    names = {BOS: "BOS", STORE: "STORE", QUERY: "QUERY", READOUT: "READOUT"}
    out = []
    for t in ids.tolist():
        if t in names:
            out.append(names[t])
        elif KEY0 <= t < VAL0:
            out.append(f"k{t - KEY0}")
        elif VAL0 <= t < FILL0:
            out.append(f"v{t - VAL0}")
        else:
            out.append(f"_{t - FILL0}")          # inert filler
    return " ".join(out)
