import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def remove_outlier(df, col, iqr_factor=4, na_rm=False, directions="both"):
    col_data = df[col]
    if na_rm:
        q25, q75 = np.nanpercentile(col_data, [25 ,75])
    else:
        q25, q75 = np.percentile(col_data, [25 ,75])
        
    iqr = q75 - q25
    
    if directions == "both":
        return df.query(f"({col} > ( {q25} - {iqr_factor}*{iqr})) & ({col} < ({q75} + {iqr_factor}*{iqr}))")
    
    if directions == "upper":
        return df.query(f"{col} < ( {q75} + {iqr_factor}*{iqr}) ")
    
    if directions == "lower":
        return df.query(f"{col} > ( {q25} - {iqr_factor}*{iqr}) ")
    
    else:
        raise ValueError("direction must be one of 'both', 'upper', or 'lower'")
    
    
def get_iqr_outliers(df, col, iqr_factor=4, na_rm=False, directions="both"):
    col_data = df[col]
    if na_rm:
        q25, q75 = np.nanpercentile(col_data, [25 ,75])
    else:
        q25, q75 = np.percentile(col_data, [25 ,75])
        
    iqr = q75 - q25
    
    if directions == "both":
        return df.query(f"({col} <= ( {q25} - {iqr_factor}*{iqr})) | ({col} >= ({q75} + {iqr_factor}*{iqr}))")
    
    if directions == "upper":
        return df.query(f"{col} >= ( {q75} + {iqr_factor}*{iqr}) ")
    
    if directions == "lower":
        return df.query(f"{col} <= ( {q25} - {iqr_factor}*{iqr}) ")
    
    else:
        raise ValueError("direction must be one of 'both', 'upper', or 'lower'")

        
def standardize(x, center=True):
    if center:
        ctr = x.mean()
    else:
        ctr = 0
    return (x - ctr ) / np.std(x)

def destandardize_area(area_s):
    return area_s*np.std(d["living_space"]) + d["living_space"].mean()

def destandardize_price(price_s):
    return price_s * 100000

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

