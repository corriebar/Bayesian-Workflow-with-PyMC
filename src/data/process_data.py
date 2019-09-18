import pandas as pd
import numpy as np
from src.utils.utils import remove_outlier

# TODO make filepaths nicer
d = pd.read_csv("../../data/raw_data/houses.csv", dtype = {"ags": str, "plz": str}, index_col=0)

# TODO maybe make functions out of this
# process features

d["price"] = d["herstellung"].fillna(0) + d["grundstueckskaufpreis"].fillna(0) + d["kaufpreis"].fillna(0) + d["modernisierung"].fillna(0)
d["living_space"] = d["gesamtwohnflaeche"]
d["angenommendatum"] = pd.to_datetime(d["angenommendatum"])
d["sale_year"] = d["angenommendatum"].dt.year
d["sale_month"] = d["angenommendatum"].dt.month

d["sqm_price"] = d["price"] / d["living_space"]
d["log_sqm_price"] = np.log(d["sqm_price"])

d["const_year"] = np.where(d["baujahr"].isna() & d["verwendungszweck"].isin(["NEUBAU", "KAUF_NEUBAU_VOM_BAUTRAEGER"]),
                          d["sale_year"], d["baujahr"])

# translate features
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


#### filter data
d = d.query("objecttype != 'empty_lot'") \
    .query("erfolgreich == 1")       \
    .query("living_space > 0")


d = remove_outlier(d, col="log_sqm_price", iqr_factor=2.5)


# select columns
d = d[['ags', 'zip', 'lat', 'lng', 'price', 'sqm_price', 'log_sqm_price', 'living_space', 'sale_year',
       'sale_month', 'const_year',  'objecttype', 'housetype', 'usage',
       'interior_qual']]


# save data frame
d.to_csv("../../data/interim_data/houses.csv")