"""
stat_tests.py
=============
TrialStat AI — Statistical test computation module.

decision_engine.py decides *which* test to use.
This module *runs* the selected test and returns the results.

All four methods are implemented, including Bootstrap resampling —
a modern, distribution-free approach that is especially robust
for small or irregularly distributed samples.

实际执行统计检验计算。

decision_engine.py 负责"该用哪个检验"，这个模块负责"把检验真正跑出来"。
四种方法都实现了，包括用户特别要求的 Bootstrap 重抽样法
（在小样本/非正态情况下，比传统检验更稳健）。

"""

import numpy as np
from scipy import stats


def run_independent_t_test(group_a, group_b):
    """Standard independent samples t-test (assumes equal variances)."""
    stat, p = stats.ttest_ind(group_a, group_b, equal_var=True)
    return {
        "test_name":   "Independent Samples t-test",
        "statistic":   round(float(stat), 4),
        "p_value":     round(float(p), 4),
        "significant": bool(p < 0.05),
    }


def run_welch_t_test(group_a, group_b):
    """Welch's t-test (does not assume equal variances)."""
    stat, p = stats.ttest_ind(group_a, group_b, equal_var=False)
    return {
        "test_name":   "Welch's t-test",
        "statistic":   round(float(stat), 4),
        "p_value":     round(float(p), 4),
        "significant": bool(p < 0.05),
    }


def run_mann_whitney(group_a, group_b):
    """Mann-Whitney U test (non-parametric, rank-based)."""
    stat, p = stats.mannwhitneyu(group_a, group_b, alternative="two-sided")
    return {
        "test_name":   "Mann-Whitney U test",
        "statistic":   round(float(stat), 4),
        "p_value":     round(float(p), 4),
        "significant": bool(p < 0.05),
    }


def run_bootstrap_test(group_a, group_b, n_bootstrap=10000, seed=42):
    """
    Bootstrap permutation test.

    Simulates the null distribution by randomly reassigning observations
    to groups n_bootstrap times, then measures how extreme the observed
    difference is within that distribution.

    Makes no distributional assumptions; highly robust for small,
    irregular samples.
    """
    group_a   = np.asarray(group_a, dtype=float)
    group_b   = np.asarray(group_b, dtype=float)
    n_a       = len(group_a)
    observed  = float(np.mean(group_a) - np.mean(group_b))
    combined  = np.concatenate([group_a, group_b])

    rng   = np.random.default_rng(seed)
    diffs = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        perm      = rng.permutation(combined)
        diffs[i]  = np.mean(perm[:n_a]) - np.mean(perm[n_a:])

    p_value = float(np.mean(np.abs(diffs) >= np.abs(observed)))

    return {
        "test_name":   "Bootstrap Resampling (Permutation test)",
        "statistic":   round(observed, 4),
        "p_value":     round(p_value, 4),
        "significant": bool(p_value < 0.05),
        "n_bootstrap": n_bootstrap,
    }


# Map test identifier -> function; used by run_test() and app.py
TEST_FUNCTIONS = {
    "independent_t_test": run_independent_t_test,
    "welch_t_test":       run_welch_t_test,
    "mann_whitney_u":     run_mann_whitney,
    "bootstrap_t_test":   run_bootstrap_test,
}


def run_test(test_name, group_a, group_b):
    """Unified entry point: run the named test on the provided data."""
    if test_name not in TEST_FUNCTIONS:
        raise ValueError(f"Unknown test: {test_name}")
    return TEST_FUNCTIONS[test_name](group_a, group_b)


if __name__ == "__main__":
    # Run all four tests on the goat demo data for comparison
    control   = [22.5, 23, 31.5, 23.5]
    treatment = [21.5, 0.75, 4.3, 30, 3, 28.5, 11.5, 24.5]

    print("=== All four tests on the same dataset ===\n")
    for name in TEST_FUNCTIONS:
        r = run_test(name, control, treatment)
        print(f"{r['test_name']}: p = {r['p_value']}, significant: {r['significant']}")
