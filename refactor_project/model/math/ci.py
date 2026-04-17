import math

import numpy as np
from scipy.stats import t as tdist


def mean_ci95(x):
    x = np.array(x, dtype=float)
    m = float(x.mean())
    s = float(x.std(ddof=1)) if len(x) > 1 else 0.0
    half = 1.96 * s / max(np.sqrt(len(x)), 1.0)
    return m, (m - half), (m + half)


def mean_ci_t(x, alpha: float = 0.05):
    """
    Média e IC (1-alpha) via distribuição t de Student.
    Correto para n pequeno (n=5 seeds típico neste projeto).
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    m = float(x.mean())
    if n <= 1:
        return m, m, m
    se = float(x.std(ddof=1)) / math.sqrt(n)
    half = float(tdist.ppf(1.0 - alpha / 2.0, df=n - 1)) * se
    return m, m - half, m + half


def mean_ci_bootstrap(x, B=2000, alpha=0.05, seed=0):
    """
    Publication option: bootstrap CI of the mean.
    """
    rng = np.random.default_rng(int(seed))
    x = np.array(x, dtype=float)
    n = int(len(x))
    if n <= 1:
        m = float(x.mean()) if n == 1 else 0.0
        return m, m, m
    means = []
    for _ in range(int(B)):
        samp = rng.choice(x, size=n, replace=True)
        means.append(float(np.mean(samp)))
    means = np.array(means, dtype=float)
    lo = float(np.quantile(means, alpha / 2.0))
    hi = float(np.quantile(means, 1.0 - alpha / 2.0))
    m = float(np.mean(x))
    return m, lo, hi
