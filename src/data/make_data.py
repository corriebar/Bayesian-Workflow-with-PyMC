import requests
import zipfile
import io

def download_shapefiles():
    zip_file_url = "https://www.suche-postleitzahl.org/download_files/public/plz-gebiete.shp.zip"
    r = requests.get(zip_file_url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall("../../data/shapefiles/plz-gebiete")

def unzip_immo_data():
    with zipfile.ZipFile("../../data/raw_data/immo_data.zip") as zip_ref:
        zip_ref.extractall("../../data/raw_data")

download_shapefiles()

unzip_immo_data()