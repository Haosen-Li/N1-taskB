"""Checks for paired inference on the submitted experiment records."""
from analyze import load, paired, paired_pass_test


def test_censoring_does_not_order_unobserved_pass_times():
    # A censored at 3000 may have passed before B's observed pass at 5000.
    a = dict(t_pass=[None, 1000], cap=3000, start=0)
    b = dict(t_pass=[5000, 2000], cap=12000, start=0)
    assert paired(a, b) == (1, 0, 1)


def test_pooled_pass_comparison_uses_discordant_seed_pairs():
    a = load("X6_N192_far_answeronly")["t_pass"] + load("X7b_rep_N192_far_answeronly")["t_pass"]
    b = load("S1_N192_far")["t_pass"] + load("X7b_rep_N192_far")["t_pass"]
    result = paired_pass_test(a, b)
    assert (result["only_a"], result["only_b"]) == (13, 2)
    assert abs(result["p"] - 0.00738525390625) < 1e-12
