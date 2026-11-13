"""Mutation check of the Phase 7 and Phase 8 modules and of the follow-up code
(docs/FOLLOW_UPS.md item 6). Run with: ``uv run python scripts/mutation_followup.py``
(about ten minutes; pass a path fragment such as ``hedging`` to run one module).

Each entry replaces one line of a source file with a wrong version, runs the tests of that
module and expects them to fail ("red"); the file is restored whatever happens. A mutation that
leaves the tests green is printed as SURVIVED: either the tests cannot see that defect, or the
two versions are equivalent. **Do not interrupt it**: an interrupted run can leave one mutated
file behind (``git diff src`` shows it).
"""

import subprocess
import sys

from etfcorr.config import find_repo_root

ROOT = find_repo_root()
SRC = ROOT / "src" / "etfcorr"
U, L = "tests/unit/", "tests/leakage/"
TESTS = {
    "models/graphs.py": [U + "test_graphs.py", L + "test_graphs_no_lookahead.py"],
    "evaluation/final.py": [U + "test_eval_final.py"],
    "evaluation/portfolio.py": [U + "test_portfolio.py", U + "test_econ.py"],
    "evaluation/hedging.py": [U + "test_hedging.py", U + "test_econ.py"],
    "evaluation/econ.py": [U + "test_econ.py"],
    "evaluation/null_benchmark.py": [U + "test_null_and_scoreboard.py"],
    "evaluation/scoreboard.py": [U + "test_null_and_scoreboard.py"],
    "correlation/dcc.py": [U + "test_dcc_var_next.py"],
    "models/baselines.py": [U + "test_models.py"],
    "correlation/network_stability.py": [U + "test_network_stability.py"],
}
G, F, P, H, E = (
    "models/graphs.py",
    "evaluation/final.py",
    "evaluation/portfolio.py",
    "evaluation/hedging.py",
    "evaluation/econ.py",
)
NB, S = "evaluation/null_benchmark.py", "evaluation/scoreboard.py"
BONF = 'r["mean_diff"] < 0 and r["p_one_sided"] * len(challengers) < alpha'
DELTA = "delta = m[stress].var(ddof=1) / m[calm].var(ddof=1) - 1.0"
BAND = "h = n ** (-1.0 / 3.0) * lam[None, :]"
MUTATIONS: list[tuple[str, str, str]] = [
    (G, "np.argsort(-s, axis=1", "np.argsort(s, axis=1"),
    (G, "adj = sim >= cut", "adj = sim > cut"),
    (G, "np.argsort(-sim[i, j]", "np.argsort(sim[i, j]"),
    (G, "if not 1 <= k < n:", "if not 1 <= k <= n:"),
    (G, "F.fisher_z(corr[:, ~edge].mean(axis=1))", "F.fisher_z(corr[:, edge].mean(axis=1))"),
    (G, "if n_edges in (0, len(i)):", "if n_edges in (0,):"),
    (G, "self.similarity_window)).mean(axis=0)", "self.similarity_window))[-1]"),
    (
        F,
        "means.idxmax() if result.task.higher_is_better else means.idxmin()",
        "means.idxmin() if result.task.higher_is_better else means.idxmax()",
    ),
    (F, BONF, 'r["mean_diff"] < 0 and r["p_one_sided"] < alpha'),
    (F, BONF, BONF.replace("< 0", "> 0")),
    (
        F,
        'float(min(1.0, r["p_one_sided"] * len(challengers)))',
        'float(r["p_one_sided"] * len(challengers))',
    ),
    (F, '"h3_supported": any(', '"h3_supported": all('),
    (F, '"p_for_holm": min(', '"p_for_holm": max('),
    (F, "if marker.exists():", "if False:"),
    (F, "    if dirty:\n", "    if False:\n"),
    (F, "if not confirm_unseal:", "if False:"),
    (F, "    if not frozen:", "    if False:"),
    (F, 'best["p_one_sided"] * len(family)', 'best["p_one_sided"] * 1'),
    (P, "source.loc[source.index < date]", "source.loc[source.index <= date]"),
    (
        P,
        "forecast = dcc_corr[int(index.searchsorted(date)) - 1]",
        "forecast = dcc_corr[int(index.searchsorted(date))]",
    ),
    (
        P,
        "variances = dcc_var[int(index.searchsorted(date)) - 1]",
        "variances = dcc_var[int(index.searchsorted(date))]",
    ),
    (P, "w = w * (1.0 + row) / (1.0 + out[k])", "w = w * 1.0"),
    (P, "np.abs(target - w).sum()", "np.abs(target + w).sum()"),
    (P, "p_value = float((1 + (draws >= 0).sum())", "p_value = float((1 + (draws <= 0).sum())"),
    (P, "bool(point < 0 and p_value < alpha)", "bool(point > 0 and p_value < alpha)"),
    (P, "bool(v_point < 0 and upper < 0)", "bool(v_point < 0)"),
    (P, "alpha = 1.0 - v_left / (v_left + v_right)", "alpha = v_left / (v_left + v_right)"),
    (P, "run.turnover * bps / 1e4", "run.turnover * bps / 1e2"),
    (P, "return np.asarray(raw / raw.sum())", "return np.asarray(raw / np.abs(raw).sum())"),
    (P, BAND, BAND.replace("lam[None, :]", "lam[:, None]")),
    (P, BAND, BAND.replace("3.0", "2.0")),
    (P, "(1.0 - c - np.pi * c * lam * hilbert)", "(1.0 - c + np.pi * c * lam * hilbert)"),
    (P, "density = (0.75 / np.sqrt(5.0))", "density = (0.5 / np.sqrt(5.0))"),
    (P, "        sd = np.sqrt(dcc_var)", "        sd = dcc_var"),
    (P, 'groupby(inside.to_period("M")).first()', 'groupby(inside.to_period("M")).last()'),
    (P, 'linkage(condensed, method="single"', 'linkage(condensed, method="average"'),
    (H, "beta.shift(1).mul(hedge, axis=0)", "beta.shift(0).mul(hedge, axis=0)"),
    (H, "past = index < date", "past = index <= date"),
    (H, "past = assets.index < before", "past = assets.index <= before"),
    (H, "stress = level > np.quantile", "stress = level < np.quantile"),
    (H, "calm = level <= np.quantile", "calm = level >= np.quantile"),
    (H, "float((1 + (d >= 0).sum())", "float((1 + (d <= 0).sum())"),
    (H, 'bool(primary["diff"] < 0 and', 'bool(primary["diff"] > 0 and'),
    (H, "np.asarray(1.0 - hedged[mask]", "np.asarray(1.0 + hedged[mask]"),
    (H, DELTA, DELTA.replace("- 1.0", "- 0.0")),
    (H, DELTA, "delta = m[calm].var(ddof=1) / m[stress].var(ddof=1) - 1.0"),
    (H, 'out[f"var_{label}_{name}"] = -cut', 'out[f"var_{label}_{name}"] = cut'),
    (H, "np.quantile(values, tail, axis=0)", "np.quantile(values, 1 - tail, axis=0)"),
    (H, 'mean["r2_adj_stress"] - mean["r2_calm"]', 'mean["r2_adj_stress"] - mean["r2_stress"]'),
    (
        H,
        'mean["vol_hedged_stress"] - mean["vol_hedged_calm"]',
        'mean["vol_unhedged_stress"] - mean["vol_hedged_calm"]',
    ),
    (H, 'mean["he_stress"] - mean["he_calm"]', 'mean["he_calm"] - mean["he_stress"]'),
    (H, "(hedge**2).ewm(", "(hedge.abs()).ewm("),
    (H, "min_periods=window).var(), axis=0)", "min_periods=window).std(), axis=0)"),
    (H, "len(np.unique(idx[calm[idx]]))) >= 3:", "len(np.unique(idx[calm[idx]]))) >= 0:"),
    (
        E,
        "exp.splits.test if final else exp.splits.val",
        "exp.splits.val if final else exp.splits.test",
    ),
    (
        E,
        "H.rolling_beta(daily, hedge, design.hedge_window_days)",
        "H.rolling_beta(daily, hedge, 20)",
    ),
    (E, "skip_short=not final", "skip_short=final"),
    (E, "design.frozen_on is not None,", "True,"),
    (
        E,
        "vix.loc[inside][keep], design.stress_quantile,",
        "vix.loc[inside][keep], design.calm_quantile + 0.4,",
    ),
    (NB, "(values <= loss).sum()", "(values >= loss).sum()"),
    (NB, "* np.sqrt(1.0 - phi**2)", "* 1.0"),
    (NB, "((y - pred) ** 2).reindex(dates).mean()", "((y - pred) ** 2).mean()"),
    (S, "raw = [1.0 if e.p_value is None", "raw = [0.0 if e.p_value is None"),
    (S, "bool(e.supported and p < alpha)", "bool(p < alpha)"),
    (S, 'method="holm"', 'method="bonferroni"'),
    (S, "if e.supported and e.p_value is None:", "if False:"),
    (
        "correlation/dcc.py",
        "np.array([f.alpha for f in g]) * e2",
        "np.array([f.alpha for f in g]) * e2 * 0.5",
    ),
    (
        "correlation/dcc.py",
        "e2 = (x[seg] - np.array([f.mu for f in g])) ** 2",
        "e2 = (x[seg]) ** 2",
    ),
    ("models/baselines.py", "self.column_ = min(errors", "self.column_ = max(errors"),
    (
        "correlation/network_stability.py",
        "b = hierarchical_clusters(corr_b,",
        "b = hierarchical_clusters(corr_a,",
    ),
]


def main() -> int:
    only = sys.argv[1] if len(sys.argv) > 1 else ""
    red = survived = invalid = 0
    for rel, old, new in MUTATIONS:
        if only not in rel:
            continue
        path = SRC / rel
        source = path.read_text(encoding="utf-8")
        if source.count(old) != 1:
            print(f"INVALID  {rel}: {old!r} occurs {source.count(old)} times")
            invalid += 1
            continue
        path.write_text(source.replace(old, new), encoding="utf-8")
        try:
            done = subprocess.run(
                ["uv", "run", "pytest", "-q", "-x", "-p", "no:cacheprovider", *TESTS[rel]],
                cwd=ROOT,
                capture_output=True,
                check=False,
            )
        finally:
            path.write_text(source, encoding="utf-8")
        if done.returncode == 0:
            print(f"SURVIVED {rel}: {old!r} -> {new!r}")
            survived += 1
        else:
            red += 1
    print(f"{red} red, {survived} survived, {invalid} invalid, of {red + survived + invalid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
