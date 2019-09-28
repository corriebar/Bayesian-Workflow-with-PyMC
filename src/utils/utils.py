import pandas as pd
import numpy as np


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

        
def standardize(x):
    return (x - x.mean() ) / np.std(x)

    