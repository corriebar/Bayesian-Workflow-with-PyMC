import numpy as np
import matplotlib.pyplot as plt

from utils.data_utils import destandardize_area, destandardize_price

def set_plot_defaults(font="Europace Sans"):
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = font
    plt.rcParams['font.size'] = 18
    plt.rcParams['ytick.labelsize'] = 14
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['figure.figsize'] = 9, 9
    plt.rcParams['legend.fontsize'] = 18


def draw_model_plot(sample, draws=50, title=""):
    area_s = np.linspace(start=-1.5, stop=3, num=50)
    draws = np.random.choice(len(sample["alpha"]), replace=False, size=draws)
    alpha = sample["alpha"][draws]
    beta = sample["beta"][draws]

    mu = alpha + beta * area_s[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(13, 13))
    ax.plot(destandardize_area(area_s), destandardize_price(mu), c="gray", alpha=0.5)
    ax.set_xlabel("Living Area (in sqm)")
    ax.set_ylabel("Price (in €)")
    ax.axhline(y=8.5e6, c="#537d91", label="Most expensive flat sold in Berlin", lw=3)
    ax.axhline(y=0, c="#f77754", label="0 €", lw=3)
    ax.legend(frameon=False)
    ax.set_title(title, fontdict={'fontsize': 30})
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
    return fig, ax

