import numpy as np
import matplotlib.pyplot as plt

from .utils import _check_is_fitted


def evaluate(pricer, X, y, verbose=False):
    """
    Compute error metrics between model prices and market prices.

    Parameters
    ----------
    pricer : fitted BasePricer instance
    X : array-like (n_samples, 2) — [strike, maturity]
    y : array-like (n_samples,)  — market prices
    verbose : bool, default=False
        Print the metrics table.

    Returns
    -------
    dict with keys SSE, MSE, RMSE, NRMSE
    """
    _check_is_fitted(pricer)
    y = np.asarray(y, dtype=float)
    estimates = pricer.predict(np.asarray(X, dtype=float))

    n = len(estimates)
    SSE = float(((estimates - y) ** 2).sum())
    MSE = SSE / n
    RMSE = float(np.sqrt(MSE))
    NRMSE = float(RMSE / y.mean())

    metrics = {'SSE': SSE, 'MSE': MSE, 'RMSE': RMSE, 'NRMSE': NRMSE}

    if verbose:
        print(f"\n{'='*40}")
        print(f"{'Error Metric':<15} | {'Value':>15}")
        print(f"{'-'*40}")
        for name, val in metrics.items():
            print(f"{name:<15} | {val:>15.4f}")
        print(f"{'='*40}")

    return metrics


def plot_fit(pricer, X, y):
    """
    Plot model vs market prices with a residual panel.

    Parameters
    ----------
    pricer : fitted BasePricer instance
    X : array-like (n_samples, 2) — [strike, maturity]
    y : array-like (n_samples,)  — market prices
    """
    _check_is_fitted(pricer)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    strikes = X[:, 0]
    estimates = pricer.predict(X)

    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm",
                          "axes.spines.top": False, "axes.spines.right": False})

    fig, (ax_main, ax_res) = plt.subplots(
        2, 1, figsize=(9, 8),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )

    ax_main.scatter(strikes, y, facecolors="none", edgecolors="black", s=55, label="Market Prices")
    ax_main.scatter(strikes, estimates, color="black", marker="+", s=75, label="Model Prices")
    ax_main.set_ylabel(r"Option Price $C(K)$")
    ax_main.set_title(rf"Calibration Fit — {type(pricer).__name__}", pad=10)
    ax_main.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
    ax_main.legend(frameon=False)

    ax_res.scatter(strikes, estimates - y, color="black", s=25, alpha=0.7)
    ax_res.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax_res.set_xlabel(r"Strike $K$")
    ax_res.set_ylabel(r"Residual $\hat{C} - C$")
    ax_res.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

    plt.tight_layout()
    plt.show()


def plot_paths(paths, T, n_display=50):
    """
    Visualise simulated price paths and realised variance distribution.

    Parameters
    ----------
    paths : ndarray (n_paths, steps + 1)
    T : float — time horizon in years
    n_display : int, default=50
        Number of individual paths to draw.
    """
    n_paths = paths.shape[0]
    n = paths.shape[1] - 1
    t_grid = np.linspace(0, T, n + 1)
    idx = np.random.choice(n_paths, size=min(n_display, n_paths), replace=False)
    log_ret = np.log(paths[:, 1:] / paths[:, :-1])
    real_var = (252 / n) * np.sum(log_ret ** 2, axis=1)
    K_var = real_var.mean()

    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm",
                          "axes.spines.top": False, "axes.spines.right": False})

    fig, (ax_p, ax_h) = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"wspace": 0.35})

    for i in idx:
        ax_p.plot(t_grid, paths[i], color="black", alpha=0.4, linewidth=0.6)
    ax_p.fill_between(t_grid,
                      np.percentile(paths, 5, axis=0),
                      np.percentile(paths, 95, axis=0),
                      color="black", alpha=0.08, label="5%–95% band")
    ax_p.plot(t_grid, paths.mean(axis=0), color="black", linewidth=1.8, label="Mean path")
    ax_p.set_xlabel(r"Time $t$ (years)")
    ax_p.set_ylabel(r"Asset price $S_t$")
    ax_p.set_title(rf"Simulated Price Paths ($N={n_paths:,}$)", pad=10)
    ax_p.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
    ax_p.legend(frameon=False)

    ax_h.hist(real_var, bins=60, color="black", alpha=0.20, edgecolor="black", linewidth=0.4)
    ax_h.axvline(K_var, color="black", linewidth=1.6, label=rf"$\hat{{K}}_{{var}}={K_var:.4f}$")
    ci = 1.96 * real_var.std() / np.sqrt(n_paths)
    ax_h.axvline(K_var - ci, color="black", linewidth=0.8, linestyle="--")
    ax_h.axvline(K_var + ci, color="black", linewidth=0.8, linestyle="--", label="95% CI")
    ax_h.set_xlabel(r"Realised variance $\sigma_R^2$")
    ax_h.set_ylabel("Frequency")
    ax_h.set_title(rf"Distribution of Realised Variance ($N={n_paths:,}$)", pad=10)
    ax_h.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
    ax_h.legend(frameon=False)

    plt.tight_layout()
    plt.show()
