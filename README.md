# Bayesian Workflow with PyMC and ArviZ

Repository with code and notebook for my talk at PyConDE &amp; PyData Berlin 2019




# Setup



To install all the required dependencies from Pipfile, run
```
cd Bayesian-Workflow-with-PyMC
pipenv install
```
(If you don't have pipenv, see how to install it [here](http://docs.pipenv.org/en/latest/install/#installing-pipenv).)

To activate the environment and start the notebooks from it, run
```
pipenv shell
python -m ipykernel install --user --name=$(basename $(pwd))
jupyter lab
# or jupyter notebook
```
Then, inside jupyter, pick the according kernel for the notebooks.