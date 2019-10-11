import os
from pathlib import Path

import requests
import zipfile
import io

def download_shapefiles(path_to_extract):
    zip_file_url = "https://www.suche-postleitzahl.org/download_files/public/plz-gebiete.shp.zip"
    r = requests.get(zip_file_url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(path_to_extract)

def unzip_immo_data(path):
    with zipfile.ZipFile(os.path.join(path, "immo_data.zip")) as zip_ref:
        zip_ref.extractall("../../data/raw_data")


if __name__ == '__main__':

    project_dir = Path(__file__).resolve().parents[2]
    shapefile_path = os.path.join(project_dir, "data", "shapefiles", "plz-gebiete")
    download_shapefiles(shapefile_path)

    data_path = os.path.join(project_dir, "data", "raw_data")
    unzip_immo_data(data_path)


