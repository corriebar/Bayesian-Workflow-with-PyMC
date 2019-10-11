# Bayesian Workflow with PyMC and ArviZ

Repository with code and notebook for my talk at PyConDE &amp; PyData Berlin 2019




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

## Data
