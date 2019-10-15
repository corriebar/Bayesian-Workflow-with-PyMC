# Bayesian Workflow with PyMC and ArviZ

Repository with code and notebook for my talk at PyConDE &amp; PyData Berlin 2019.

The slides for the talk can be found [here](https://www.slideshare.net/CorrieBartelheimer/bayesian-workflow-with-pymc3-and-arviz).




# Setup


I'm using [pipenv](http://docs.pipenv.org/en/latest/install/#installing-pipenv) with Python 3.7 (the code might also work with other Python 3 versions but then you'll need to change the version in the Pipfile).
To install pipenv, run
```
pip install pipenv
```
Then install the necessary packages, using
```
cd Bayesian-Workflow-with-PyMC
pipenv install
```
To activate the environment and start the notebooks from it, run
```
pipenv shell
python -m ipykernel install --user --name=$(basename $(pwd))
jupyter lab
# or jupyter notebook
```
Then, inside jupyter, pick the according kernel for the notebooks.

## Preprocessing
To download the shapefiles and preprocess the data, run
```
make data
```

## Data
The data used in the notebooks and for the talk is by [Europace AG](https://neu.europace.de/) and not in the Repository. I instead included a data set of rental offers that I scraped from Immoscout24. A more detailed description of how I scraped the data etc can be found [here](https://www.kaggle.com/corrieaar/apartment-rental-offers-in-germany).
To use the rental data in the notebooks, you can change 
```python
d, zip_lookup, num_zip_codes = load_data(kind="prices")   
```
to 
```python
d, zip_lookup, num_zip_codes = load_data(kind="rents")
```
