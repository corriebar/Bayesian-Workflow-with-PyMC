import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.data_utils import load_data

d = load_data()

def plot_ppc_hist(inf_data):
    n = 3
    pp = inf_data.posterior_predictive
    pp = pp.stack(samples=["chain", "draw"]).reset_index("samples")
    n_draws = pp.draw.shape[0]
    draws = np.random.choice(n_draws, size=n)

    y_rep = pp.sel(samples=draws).y
    y = inf_data.observed_data.y
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 13))

    for row in range(axes.shape[0]):
        for col in range(axes.shape[1]):
            if row == 0 and col == 0:
                axes[0, 0].hist(y, bins=70, color="#3d5167", ec="#3d5167", alpha=0.9, label="$y$")
                axes[row, col].set_xlim(-3, 20)
                axes[row, col].spines["top"].set_visible(False)
                axes[row, col].spines["right"].set_visible(False)
                axes[row, col].spines["left"].set_visible(False)
                axes[row, col].get_yaxis().set_visible(False)
            else:
                draw = 2 * row + col - 1
                axes[row, col].hist(y_rep[:, draw], bins=70, color="#3d5167", alpha=0.3, ec="#3d5167", label="$y_{pred}$")
                axes[row, col].set_xlim(-3, 20)
                axes[row, col].spines["top"].set_visible(False)
                axes[row, col].spines["right"].set_visible(False)
                axes[row, col].spines["left"].set_visible(False)
                axes[row, col].get_yaxis().set_visible(False)
                # axes[row, col].set_title(f"Draw {draw +1}")
    h1, l1 = axes[0, 0].get_legend_handles_labels()
    h2, l2 = axes[1, 0].get_legend_handles_labels()
    axes[1, 1].legend(h1 + h2, l1 + l2, frameon=False, bbox_to_anchor=(1.4, 1.3))
    fig.suptitle("Observed data vs Posterior predictive")
    return fig, axes

def plot_ppc_dens(inf_data, n=20):
    pp = inf_data.posterior_predictive
    pp = pp.stack(samples=["chain", "draw"]).reset_index("samples")
    n_draws = pp.draw.shape[0]
    draws = np.random.choice(n_draws, size=n)


    y_rep = pp.sel(samples=draws).y
    y = inf_data.observed_data.y
    fig, ax = plt.subplots()
    fig.set_size_inches(15, 9)
    ax.set_xlim(-4, 20)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.get_yaxis().set_visible(False)
    sns.kdeplot(y, ax=ax, color="#3d5167", label="$y$")

    for i in range(n-1):
        sns.kdeplot(y_rep[:,i], ax=ax, color="#3d5167", alpha=0.1, label="")
    sns.kdeplot(y_rep[:, n-1], ax=ax, color="#3d5167", alpha=0.1, label="$y_{rep}$")
    ax.set_xlabel("")
    leg = ax.legend(frameon=False)
    leg.legendHandles[1].set_alpha(0.4)
    fig.suptitle("Observed data vs Posterior predictive")
    return fig, ax


def plot_ppc_scatter(inf_data):

    pp = inf_data.posterior_predictive
    pp = pp.stack(samples=["chain", "draw"]).reset_index("samples")

    y_rep_mean = pp.y.groupby("y_dim_0").mean(dim=["samples"])
    y = inf_data.observed_data.y
    fig, ax = plt.subplots()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.plot([0,50], [0,50], color="#737373")
    ax.scatter(y_rep_mean, y, s=5, alpha=0.3, color="#3d5167")
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 18)
    ax.set_xlabel("Average $y_{rep}$", labelpad=15)
    ax.set_ylabel("$y$", labelpad=15)
    fig.suptitle("Observations vs average simulated value")
    return fig, ax

def plot_ppc(inf_data, kind="density", n=None):
    if kind == "density":
        return plot_ppc_dens(inf_data, n=n)
    if kind == "hist":
        return plot_ppc_hist(inf_data)
    if kind == "scatter":
        return plot_ppc_scatter(inf_data)