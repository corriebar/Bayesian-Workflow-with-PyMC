import os
from pathlib import Path

import pandas as pd
import numpy as np

# TODO make filepaths nicer

def process_immoscout_data():
    project_dir = Path(__file__).resolve().parents[2]
    data_path = os.path.join(project_dir, "data", "raw_data", "immo_data.csv")
    immo_data_exists = os.path.isfile(data_path)

    if not immo_data_exists:
        raise Exception("Immoscout csv-File not found, download if necessary https://www.kaggle.com/corrieaar/apartment-rental-offers-in-germany and unzip first")

    d = pd.read_csv(data_path, dtype= {"geo_plz": str, "scoutId": str})

    d = make_immo_rents(d)

    d = rename_immo_features(d)

    d = filter_immo_data(d)

    d = remove_outlier(d, col="log_sqm_rent", iqr_factor=3.5)

    d = standardize_rent(d)

    d = select_rent_cols(d)

    # save data frame
    d.to_csv("../../data/interim_data/rent.csv")

def make_immo_rents(d):
    d["rent"] = np.where((d["totalRent"].notnull()) | (d["totalRent"] == 0), d["baseRent"],
                              d["totalRent"])
    d["living_space"] = d["livingSpace"]

    d["log_rent"] = np.log(d["rent"])

    d["sqm_rent"] = d["rent"] / d["living_space"]
    d["log_sqm_rent"] = np.log(d["sqm_rent"])

    d["offer_year"] = np.where(d["date"] == "Sep18", 2018, 2019)

    d["const_year"] = d["yearConstructed"]

    return d

def rename_immo_features(d):
    d["zip"] = d["geo_plz"]
    d["flattype"] = d["typeOfFlat"]
    d["interior_qual"] = d["interiorQual"]
    return d

def filter_immo_data(d):
    d = d.query("regio1 == 'Berlin'") \
        .query("rent > 0") \
        .query("living_space > 0")
    return d

def standardize_rent(d):
    # standardize log-price variables
    d["rent_s"] = d["rent"] / 100
    d["living_space_s"] = standardize(d["living_space"])
    d["log_rent_s"] =  standardize(d["log_rent"], center=True)
    d["log_sqm_rent_s"] = standardize(d["log_sqm_rent"], center=True)
    d["const_year_s"] = standardize(d["const_year"])
    return d

def select_rent_cols(d):
    # select columns
    d = d[['zip',
           'rent', 'rent_s', 'log_rent', 'log_rent_s',
           'sqm_rent', 'log_sqm_rent', 'log_sqm_rent_s',
           'living_space', 'living_space_s', 'offer_year',
           'const_year', 'const_year_s', 'flattype',
           'interior_qual']]
    return d


def process_europace_data():

    project_dir = Path(__file__).resolve().parents[2]
    data_path = os.path.join(project_dir, "data", "raw_data", "houses.csv")
    houses_data_exists = os.path.isfile(data_path)

    if not houses_data_exists:
        return print("Data not available, use Immoscout-Data instead")

    d = pd.read_csv("../../data/raw_data/houses.csv", dtype = {"ags": str, "plz": str}, index_col=0)

    d = make_ep_prices(d)

    d = translate_ep_features(d)

    d = filter_data(d)

    d = remove_outlier(d, col="log_sqm_price", iqr_factor=3.5)

    d = standardize_df(d)

    d = select_cols(d)
    # save data frame
    d.to_csv("../../data/interim_data/houses.csv")

def make_ep_prices(d):
    d["price"] = d["herstellung"].fillna(0) + d["grundstueckskaufpreis"].fillna(0) + d["kaufpreis"].fillna(0) + d["modernisierung"].fillna(0)
    d["living_space"] = d["gesamtwohnflaeche"]
    d["angenommendatum"] = pd.to_datetime(d["angenommendatum"])
    d["sale_year"] = d["angenommendatum"].dt.year
    d["sale_month"] = d["angenommendatum"].dt.month

    d["log_price"] = np.log(d["price"])

    d["sqm_price"] = d["price"] / d["living_space"]
    d["log_sqm_price"] = np.log(d["sqm_price"])

    d["const_year"] = np.where(d["baujahr"].isna() & d["verwendungszweck"].isin(["NEUBAU", "KAUF_NEUBAU_VOM_BAUTRAEGER"]),
                              d["sale_year"], d["baujahr"])

    return d

    # translate features

def translate_ep_features(d):
    d["zip"] = d["plz"]
    d["objecttype"] = d["objektart"].replace({"EIGENTUMSWOHNUNG": "flat", "HAUS":"house", "GRUNDSTUECK": "empty_lot" })
    d["housetype"] = d["haustyp"].replace({"EINFAMILIENHAUS": "single_family_house",
                                          "DOPPELHAUSHAELFTE": "semidetached_house",
                                          "REIHENHAUS": "terrace_house",
                                          "MEHRFAMILIENHAUS": "multi_family_house",
                                          "ZWEIFAMILIENHAUS": "two_family_house"})
    d["usage"] = d["nutzungsart"].replace({"EIGENGENUTZT": "owner_occupied",
                                          "VERMIETET": "rent_out",
                                          "TEIL_VERMIETET": "partly_rent_out"})
    d["interior_qual"] = d["ausstattung"].replace({"STARK_GEHOBEN": "luxury",
                                             "GEHOBEN": "sophisticated",
                                             "MITTEL": "normal",
                                             "EINFACH": "basic"})
    return d


    #### filter data

def filter_data(d):
    d = d.query("objecttype != 'empty_lot'") \
        .query("erfolgreich == 1")       \
        .query("living_space > 0")
    return d

def standardize_df(d):
    # standardize log-price variables
    d["price_s"] = d["price"] / 100000
    d["living_space_s"] = standardize(d["living_space"])
    d["log_price_s"] =  standardize(d["log_price"], center=True)
    d["log_sqm_price_s"] = standardize(d["log_sqm_price"], center=True)
    d["const_year_s"] = standardize(d["const_year"])

    return d

def select_cols(d):
    # select columns
    d = d[['ags', 'zip', 'lat', 'lng',
           'price', 'price_s', 'log_price', 'log_price_s',
           'sqm_price', 'log_sqm_price', 'log_sqm_price_s',
           'living_space', 'living_space_s', 'sale_year',
           'sale_month', 'const_year',  'const_year_s',
           'objecttype', 'housetype', 'usage',
           'interior_qual']]
    return d


def remove_outlier(df, col, iqr_factor=4, na_rm=False, directions="both"):
    col_data = df[col]
    if na_rm:
        q25, q75 = np.nanpercentile(col_data, [25, 75])
    else:
        q25, q75 = np.percentile(col_data, [25, 75])

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
        q25, q75 = np.nanpercentile(col_data, [25, 75])
    else:
        q25, q75 = np.percentile(col_data, [25, 75])

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
    return (x - ctr) / np.std(x)


if __name__ == '__main__':

    process_europace_data()

    process_immoscout_data()