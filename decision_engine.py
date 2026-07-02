"""
decision_engine.py
==================
TrialStat AI — Statistical test recommendation engine.

This is the core of the project — a deterministic rule-based system (not AI)
that recommends the appropriate statistical test based on data characteristics:
normality, variance homogeneity, and sample size.

Decision logic:
  Normal distribution + equal variances          -> Independent samples t-test (highest power)
  Normal distribution + unequal variances        -> Welch's t-test (no equal-variance assumption)
  Non-normal / unknown distribution / very small -> Mann-Whitney U test (primary)
  sample size                                      Bootstrap resampling (alternative)


统计检验方法自动推荐引擎。

这是整个项目的核心——一套确定性的规则逻辑（不是AI判断），
根据数据的统计特征（正态性、方差齐性、样本量）来推荐合适的检验方法。

判断逻辑：
  正态分布 + 方差齐性               → 独立样本 t 检验（效能最高）
  正态分布 + 方差不齐               → Welch's t 检验（不需要方差齐性假设）
  非正态分布 / 分布未知 / 样本量极小 → Mann-Whitney U 检验（首选，抗造）
                                      备选：Bootstrap 重抽样检验
"""

import numpy as np
from scipy import stats


def select_test(group_a, group_b, alpha=0.05, small_sample_threshold=5):
    """
    Automatically recommend the appropriate statistical test for two independent samples.

    Parameters:
        group_a, group_b          : Two groups of numeric data (list or numpy array)
        alpha                     : Significance level, default 0.05
        small_sample_threshold    : Minimum per-group size before flagging as small sample
                                    (default 5 — below this, route conservatively to
                                    Mann-Whitney regardless of normality)

    Returns:
        recommended_test : str  — identifier of the recommended test
        diagnostics      : dict — full audit trail of every decision made
    """
    group_a = np.asarray(group_a, dtype=float)
    group_b = np.asarray(group_b, dtype=float)

    n_a, n_b = int(len(group_a)), int(len(group_b))
    diagnostics = {
        "sample_sizes": {"group_a": n_a, "group_b": n_b},
        "alpha": alpha,
    }

    # ---------- Step 1: Sample size check ----------
    is_small_sample = bool(min(n_a, n_b) < small_sample_threshold)
    diagnostics["is_small_sample"] = is_small_sample
    diagnostics["small_sample_threshold"] = small_sample_threshold

    # ---------- Step 2: Normality test (Shapiro-Wilk) ----------
    # Shapiro-Wilk requires n >= 3; handle edge case gracefully
    if n_a >= 3 and n_b >= 3:
        _, p_norm_a = stats.shapiro(group_a)
        _, p_norm_b = stats.shapiro(group_b)
        is_normal = bool((p_norm_a > alpha) and (p_norm_b > alpha))
        diagnostics["normality"] = {
            "method": "Shapiro-Wilk test",
            "group_a_p_value": round(float(p_norm_a), 4),
            "group_b_p_value": round(float(p_norm_b), 4),
            "is_normal": is_normal,
            "interpretation": (
                "Both groups are consistent with normality"
                if is_normal
                else "Normality assumption rejected in at least one group"
            ),
        }
    else:
        is_normal = False
        diagnostics["normality"] = {
            "method": "Shapiro-Wilk not applicable (n < 3)",
            "is_normal": False,
            "interpretation": "Sample too small; conservatively treated as non-normal",
        }

    # ---------- Step 3: Decision branching ----------
    if is_normal and not is_small_sample:
        # Normal + sufficient sample size -> check variance homogeneity
        _, p_levene = stats.levene(group_a, group_b)
        is_equal_var = bool(p_levene > alpha)
        diagnostics["variance_homogeneity"] = {
            "method": "Levene's test",
            "p_value": round(float(p_levene), 4),
            "is_equal_var": is_equal_var,
            "interpretation": (
                "Equal variances: spread is similar across groups"
                if is_equal_var
                else "Unequal variances: spread differs substantially between groups"
            ),
        }

        if is_equal_var:
            recommended_test = "independent_t_test"
            diagnostics["decision_path"] = (
                "Normal distribution + equal variances -> Independent samples t-test"
            )
        else:
            recommended_test = "welch_t_test"
            diagnostics["decision_path"] = (
                "Normal distribution + unequal variances -> Welch's t-test"
            )

    else:
        # Non-normal or very small sample -> non-parametric route
        recommended_test = "mann_whitney_u"
        diagnostics["alternative_test"] = "bootstrap_t_test"

        reasons = []
        if not is_normal:
            reasons.append("normality assumption violated")
        if is_small_sample:
            reasons.append(
                f"small sample (minimum group size: {min(n_a, n_b)})"
            )
        diagnostics["decision_path"] = (
            f"{' + '.join(reasons).capitalize()} "
            f"-> Mann-Whitney U test (alternative: Bootstrap resampling)"
        )

    diagnostics["recommended_test"] = recommended_test
    return recommended_test, diagnostics


# Display names for use in UI and reports
TEST_DISPLAY_NAMES = {
    "independent_t_test": "Independent Samples t-test",
    "welch_t_test":       "Welch's t-test",
    "mann_whitney_u":     "Mann-Whitney U test",
    "bootstrap_t_test":   "Bootstrap Resampling (Permutation test)",
}


if __name__ == "__main__":
    # Self-test using the goat deworming vaccine demo data
    control   = [22.5, 23, 31.5, 23.5]
    treatment = [21.5, 0.75, 4.3, 30, 3, 28.5, 11.5, 24.5]

    test_name, diag = select_test(control, treatment)
    print("Recommended test :", TEST_DISPLAY_NAMES[test_name])
    print("Decision path    :", diag["decision_path"])
    import json
    print(json.dumps(diag, indent=2, ensure_ascii=False))
