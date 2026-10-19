"""Correctness checks for the ensemble trainer and the task generator (CPU, float64)."""
import torch

from ensemble import Ensemble, EnsembleAdamW, ModelCfg
from task import TaskCfg, make_batch, relevant_edit, query_edit, STORE, QUERY, VAL0, FILL0, NV

torch.set_default_dtype(torch.float64)


def _model(seeds, **kw):
    m = Ensemble(ModelCfg(**kw), seeds, "cpu")
    for p in m.params.values():
        p.data = p.data.double()
    return m


def test_members_are_independent_and_match_single_models():
    ids = torch.randint(28, (3, 5, 40))
    ens = _model([1, 2, 3])
    out = ens(ids)
    for e, s in enumerate([1, 2, 3]):
        single = _model([s])
        torch.testing.assert_close(single(ids[e:e + 1])[0], out[e])


def test_gradients_do_not_leak_between_members():
    ens = _model([1, 2])
    ids = torch.randint(28, (2, 4, 30))
    ens(ids)[0].sum().backward()          # loss touches member 0 only
    for p in ens.params.values():
        assert p.grad[1].abs().max() == 0


def test_sdpa_equals_manual_attention_and_causality():
    ens = _model([4, 5])
    ids = torch.randint(28, (2, 3, 50))
    a = ens(ids)
    b, _ = ens(ids, attn=True)            # manual path
    torch.testing.assert_close(a, b)
    ids2 = ids.clone(); ids2[:, :, 30:] = (ids2[:, :, 30:] + 1) % 28
    torch.testing.assert_close(ens(ids2)[:, :, :30], a[:, :, :30])  # future never leaks


def test_ablation_block_is_exact_and_local():
    ens = _model([7, 8])
    ids = torch.randint(28, (2, 3, 40))
    empty = [None, torch.zeros(3, 40, 40, dtype=torch.bool)]
    torch.testing.assert_close(ens(ids, block=empty), ens(ids))
    blk = torch.zeros(3, 40, 40, dtype=torch.bool); blk[:, -1, 5:20] = True
    out = ens(ids, block=[None, blk])
    torch.testing.assert_close(out[:, :, :-1], ens(ids)[:, :, :-1])      # only the final row changes
    assert (out[:, :, -1] - ens(ids)[:, :, -1]).abs().max() > 0


def test_window_blocks_far_tokens():
    ens = _model([6], layers=2)
    ids = torch.randint(28, (1, 2, 60))
    ids2 = ids.clone(); ids2[:, :, :10] = (ids2[:, :, :10] + 3) % 28
    # 2 layers x window 8 => receptive field 14 < 59-9
    torch.testing.assert_close(ens(ids, window=8)[:, :, -1], ens(ids2, window=8)[:, :, -1], atol=0, rtol=0)


def test_per_member_clipping():
    ens = _model([1, 2])
    opt = EnsembleAdamW(ens.params.values())
    for p in ens.params.values():
        p.grad = torch.ones_like(p); p.grad[1] *= 1e-6
    norms = opt.clip(1.0)
    after = sum(p.grad.flatten(1).square().sum(1) for p in ens.params.values()).sqrt()
    assert abs(after[0] - 1) < 1e-6 and abs(after[1] - norms[1]) < 1e-12


def test_adamw_matches_reference_formula():
    torch.manual_seed(0)
    ps = [torch.nn.Parameter(torch.randn(2, 3, 4)), torch.nn.Parameter(torch.randn(2, 5))]
    ref = [p.detach().clone() for p in ps]
    opt = EnsembleAdamW(ps, lr=0.01, wd=0.1)
    m = [torch.zeros_like(r) for r in ref]; v = [torch.zeros_like(r) for r in ref]
    for t in range(1, 4):
        gs = [torch.randn_like(r) for r in ref]
        for p, g in zip(ps, gs):
            p.grad = g.clone()
        opt.step()
        for i, g in enumerate(gs):
            m[i] = .9 * m[i] + .1 * g; v[i] = .999 * v[i] + .001 * g * g
            ref[i] = ref[i] * (1 - 0.01 * 0.1) - 0.01 * (m[i] / (1 - .9 ** t)) / ((v[i] / (1 - .999 ** t)).sqrt() + 1e-8)
    for p, r in zip(ps, ref):
        torch.testing.assert_close(p.detach(), r)


def test_task_answers_and_pairing():
    for layout in ["qk", "v1direct", "v1delayed"]:
        c = TaskCfg(N=40, post_lo=0, post_hi=40, layout=layout)
        b = make_batch(300, torch.Generator().manual_seed(0), "cpu", c)
        x = b["x"]
        assert x.shape[1] == c.T
        rows = torch.arange(300)
        q = x[rows, b["qpos"]]
        # the answer is the token right after the stored key equal to the query
        for i in range(300):
            k = ((x[i] == q[i]) & (torch.arange(c.T) < b["qpos"])).nonzero()
            assert len(k) == 1 and x[i, k[0, 0] - 1] == STORE
            assert x[i, k[0, 0] + 1] == b["answer"][i] == x[i, b["src"][i]]
        assert (b["distance"] == c.T - 1 - b["src"]).all()
        assert b["distance"].min() >= (2 if layout == "qk" else 3) and b["distance"].max() <= 40 + 12
        xr, new = relevant_edit(b)
        assert ((xr != x).sum(1) == 1).all() and (new != b["answer"]).all()
        xq, newq = query_edit(b)
        assert ((xq != x).sum(1) == 1).all()
    v = make_batch(200, torch.Generator().manual_seed(3), "cpu", TaskCfg(N=64, post_lo=0, post_hi=64))
    f = make_batch(200, torch.Generator().manual_seed(3), "cpu",
                   TaskCfg(N=64, post_lo=0, post_hi=64, noise="filler"))
    m = make_batch(200, torch.Generator().manual_seed(3), "cpu",
                   TaskCfg(N=64, post_lo=0, post_hi=64, noise="mix", n_confusable=16))
    nm = v["noise_mask"]
    assert (v["x"][~nm] == f["x"][~nm]).all() and (v["x"][nm] - VAL0 == f["x"][nm] - FILL0).all()
    assert (m["confusable"].sum(1) == 16).all() and (f["x"][nm] >= FILL0).all()
    assert (v["answer"] == f["answer"]).all() and (v["answer"] == m["answer"]).all()


def test_fixed_head_is_one_hot_and_leaves_other_heads_alone():
    """X1 intervention: layer-0 head 0 looks only at t-k; everything else is the ordinary model."""
    ids = torch.randint(28, (2, 3, 30))
    plain = _model([1, 2])
    for k in (1, 2):
        fixed = _model([1, 2], fixed_head_offset=k)
        _, maps_f = fixed(ids, attn=True)
        _, maps_p = plain(ids, attn=True)
        w = maps_f[0][:, :, 0]                                   # [E,B,T,T] layer 0, head 0
        T = ids.shape[-1]
        target = torch.zeros(T, T)
        for t in range(T):
            target[t, t - k if t >= k else t] = 1
        torch.testing.assert_close(w, target.expand_as(w))
        torch.testing.assert_close(maps_f[0][:, :, 1:], maps_p[0][:, :, 1:])   # other heads untouched


def test_reinit_restores_original_init_of_one_layer_only():
    from types import SimpleNamespace
    from ensemble import init_params
    from train import load_start
    m = _model([3, 4])
    for prm in m.params.values():
        prm.data.add_(1.0)                                        # pretend it was trained
    trained = {k: v.detach().clone() for k, v in m.params.items()}
    args = SimpleNamespace(init_from=None, resume=None, reinit="l0")
    load_start(args, m, None, None, "cpu")
    fresh = init_params(m.cfg, [3, 4])
    for k, v in fresh.items():
        now = m.P(k).detach()
        if k.startswith("l0."):
            torch.testing.assert_close(now, v)                    # back to the seed's own init
        else:
            torch.testing.assert_close(now, trained[k.replace(".", "_")])


def test_reference_model_matches_ensemble_member():
    """reference_model.py (one plain model) == member e of the ensemble, for every model variant."""
    from reference_model import ReferenceTransformer
    ids = torch.randint(28, (3, 40))
    for kw in [{}, {"fixed_head_offset": 1}, {"pe": "learned", "slopes": [0.0, 0.0, 0.5, 1.0]}]:
        ens = _model([11, 12], **kw)
        out = ens(ids[None].expand(2, -1, -1))
        for e in range(2):
            params = {k.replace("_", ".", 1): v[e].detach() for k, v in ens.params.items()}
            ref = ReferenceTransformer(params, ens.cfg)
            logits, _ = ref(ids)
            torch.testing.assert_close(logits, out[e])


def test_other_record_counts():
    for R in (1, 2, 8):
        c = TaskCfg(N=20, post_lo=0, post_hi=20, n_records=R)
        b = make_batch(200, torch.Generator().manual_seed(1), "cpu", c)
        x = b["x"]
        assert x.shape[1] == c.T == 1 + 20 + 3 * R + 2 and b["vpos"].shape == (200, R)
        for i in range(200):
            k = ((x[i] == x[i, -1]) & (torch.arange(c.T) < c.T - 1)).nonzero()
            assert len(k) == 1 and x[i, k[0, 0] + 1] == b["answer"][i]
