import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_data():
    d = pd.read_csv("../data/interim_data/houses.csv", dtype={"ags": str, "zip": str}, index_col=0)
    d.zip = d.zip.map(str.strip)
    zip_codes = np.sort(d.zip.unique())
    num_zip_codes = len(zip_codes)
    zip_lookup = dict(zip(zip_codes, range(num_zip_codes)))
    d["zip_code"] = d.zip.replace(zip_lookup).values
    return d, zip_lookup, num_zip_codes

d, zip_lookup, _ = load_data()

def map_zip_codes(zip_strings):
    return pd.Series(zip_strings).replace(zip_lookup).values



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

def standardize_area(area, d):
    return (area - np.mean(d["living_space"]) ) / np.std(d["living_space"])

def destandardize_area(area_s, d=d):
    return area_s*np.std(d["living_space"]) + d["living_space"].mean()

def destandardize_price(price_s):
    return price_s * 100000



