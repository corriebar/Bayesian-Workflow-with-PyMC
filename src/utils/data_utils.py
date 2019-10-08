import os

import pandas as pd
import numpy as np
import pkg_resources


def load_data(kind="prices"):
    house_price_data_exists = os.path.isfile("../data/interim_data/houses.csv")
    if kind == "prices" and house_price_data_exists:
        d = pd.read_csv("../data/interim_data/houses.csv", dtype={"ags": str, "zip": str}, index_col=0)
    if kind == "rents":
        d = pd.read_csv("../data/interim_data/rent.csv", dtype={"zip": str}, index_col=0)
    d.zip = d.zip.map(str.strip)
    zip_codes = np.sort(d.zip.unique())
    num_zip_codes = len(zip_codes)
    zip_lookup = dict(zip(zip_codes, range(num_zip_codes)))
    d["zip_code"] = d.zip.replace(zip_lookup).values
    return d, zip_lookup, num_zip_codes

d, zip_lookup, _ = load_data()

def map_zip_codes(zip_strings, zip_lookup=zip_lookup):
    return pd.Series(zip_strings).replace(zip_lookup).values

def standardize_area(area, d):
    return (area - np.mean(d["living_space"]) ) / np.std(d["living_space"])

def destandardize_area(area_s, d=d):
    return area_s*np.std(d["living_space"]) + d["living_space"].mean()

def destandardize_price(price_s):
    return price_s * 100000



