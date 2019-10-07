import numpy as np
import matplotlib.pyplot as plt

from utils.data_utils import destandardize_area, destandardize_price, load_data

d, _, _ = load_data()

def set_plot_defaults(font="Roboto"):
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = font
    plt.rcParams['font.size'] = 18
    plt.rcParams['ytick.labelsize'] = 16
    plt.rcParams['xtick.labelsize'] = 16
    plt.rcParams['figure.figsize'] = 9, 9
    plt.rcParams['legend.fontsize'] = 18
    plt.rcParams['axes.titlesize'] = 30
    plt.rcParams['figure.titlesize'] = 30

def draw_model_plot(sample, draws=50, title=""):
    area_s = np.linspace(start=-1.5, stop=3, num=50)
    draws = np.random.choice(len(sample["alpha"]), replace=False, size=draws)
    alpha = sample["alpha"][draws]
    beta = sample["beta"][draws]

    mu = alpha + beta * area_s[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(13,13))
    ax.plot(destandardize_area(area_s, d=d), destandardize_price(mu), c="#737373", alpha=0.5)
    ax.set_xlabel("Living Area (in sqm)", fontdict={"fontsize": 22})
    ax.set_ylabel("Price (in €)",  fontdict={"fontsize": 22})
    ax.set_title(title, fontdict={'fontsize': 35})
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return fig, ax


def plot_pred_hist(y_pred, title=None, threshold=None):
    fig, ax = plt.subplots(figsize=(15, 9))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.get_yaxis().set_visible(False)
    bins = np.arange(0, 800000, 20000)
    if threshold is not None:

        ax.hist(y_pred[y_pred >= threshold], color="#3d5167", ec="#3d5167", alpha=0.8, bins=bins)
        ax.hist(y_pred[y_pred < threshold], color="#f07855", ec="#f07855", bins=bins, alpha=0.8)
        ax.annotate(s=f"Pr(Price < {threshold/1000:.0f}k) = \n{np.mean(y_pred < threshold) * 100:.1f}%", xy=(100000, 25),
                    ha="center", fontsize=22)
    else:
        ax.hist(y_pred, color="#3d5167", ec="#3d5167", alpha=0.8, bins=bins)
    ax.set_xticks([0, 100000, 300000, 500000, 700000])
    ax.set_title(title)
    return fig, ax


def plot_price_area():
    fig, ax = plt.subplots()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.ylim(0, 1500000)
    plt.xlim(0, 400)
    ax.scatter(d["living_space"], d["price"], s=1.8, c="#737373", alpha=0.3)
    ax.set_xlabel("Living Area (in sqm)")
    ax.set_ylabel("Price (in €)")
    ax.set_title("Price versus Living Area", fontdict={'fontsize': 30})
    return fig, ax
