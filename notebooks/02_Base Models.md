# Base Models
In this notebook, we'll build some base models with PyMC. 
We start with an intercept-only model (basically estimating the mean) and then extend this to a linear model.
For both models we try some different likelihoods, namely a normal likelihood as well as a Student-T likelihood (robust model).


```python
import sys
sys.path.append('../src/')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pymc3 as pm

import arviz as az

from utils.data_utils import destandardize_area, destandardize_price, load_data, standardize_area
from utils.plot_utils import set_plot_defaults
```

    WARNING (theano.tensor.blas): Using NumPy C-API based implementation for BLAS functions.
    /home/corrie/.pyenv/versions/anaconda3-2019.03/lib/python3.7/site-packages/xarray/core/merge.py:17: FutureWarning: The Panel class is removed from pandas. Accessing it from the top-level namespace will also be removed in the next version
      PANDAS_TYPES = (pd.Series, pd.DataFrame, pd.Panel)



```python
d, _, _ = load_data(kind="prices")   # loads data from data/interim_data/houses.csv 
                                     # aternatively, use kind="rents" to load data from data/interim_data/rent.csv
set_plot_defaults("Europace Sans")
target = "price_s"
```

## Intercept only model

$$\begin{align*}
price &\sim \text{Normal}(\mu, \sigma)\\
\mu &\sim \text{Normal}(0, 10)\\
\sigma &\sim \text{HalfNormal}(10)
\end{align*}$$

I'm also trying out different priors for $\sigma$. If you want to know more about which priors are recommended for which kind of model, check this [prior recommendation](https://github.com/stan-dev/stan/wiki/Prior-Choice-Recommendations) by the Stan team.


```python
with pm.Model() as intercept_normal:
    mu = pm.Normal("mu", mu=0, sd=10)
    sigma = pm.HalfNormal("sigma", sd=10)
    
    y = pm.Normal("y", mu=mu, sd=sigma, observed = d[target])
    
    trace = pm.sample(random_seed=2412, chains=4, 
                      draws=1000, tune=1000)
```

    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (4 chains in 2 jobs)
    NUTS: [sigma, mu]
    Sampling 4 chains: 100%|██████████| 8000/8000 [00:06<00:00, 1150.26draws/s]



```python
az.plot_trace(trace, coords={"chain":[0,1]})
plt.show()
```


![png](02_Base%20Models_files/02_Base%20Models_6_0.png)


The robust version of this model uses  Student-T likelihood instead:
$$\begin{align*}
price &\sim \text{Student-T}(\mu, \sigma, \nu)\\
\mu &\sim \text{Normal}(0, 10)\\
\sigma &\sim \text{HalfCauchy}(10)\\
\nu &\sim \text{Gamma}(\alpha=2, \beta=0.1)
\end{align*}$$


```python
with pm.Model() as intercept_student:
    mu = pm.Normal("mu", mu=0, sd=20)
    sigma = pm.HalfCauchy("sigma", 10)
    nu = pm.Gamma("nu", alpha=2, beta=0.1)
    
    y = pm.StudentT("y", mu=mu, sd=sigma, nu=nu, observed=d[target])
    
    trace = pm.sample(random_seed=2412, chains=4, 
                      draws=1000, tune=1000)
```

    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (4 chains in 2 jobs)
    NUTS: [nu, sigma, mu]
    Sampling 4 chains: 100%|██████████| 8000/8000 [00:16<00:00, 483.90draws/s]



```python
az.plot_trace(trace)
plt.show()
```


![png](02_Base%20Models_files/02_Base%20Models_9_0.png)


## Linear base model
We add the predictor of the living area to the model. Note that the living area is standardized, so that the intercept represents the price for an average-sized home (which is 101sqm).

$$\begin{align*}
price &\sim \text{Normal}(\mu, \sigma)\\
\mu &= \alpha + \beta \;\text{living_space}\\
\alpha &\sim \text{Normal}(0, 10) \\
\beta &\sim \text{Normal}(0, 1) \\
\sigma &\sim \text{Exponential}(1/5)
\end{align*}$$


```python
with pm.Model() as lin_normal:
    living_space = pm.Data("living_space", d["living_space_s"])
    
    alpha = pm.Normal("alpha", mu=0, sd=10)
    beta = pm.Normal("beta", mu=0, sd=1)
    mu = alpha + beta*living_space
    
    sigma = pm.Exponential("sigma", 1/5)
    y = pm.Normal("y", mu=mu, sd=sigma, observed=d["price_s"])
    trace = pm.sample(random_seed=2412, chains=4, 
                      draws=1000, tune=1000)
```

    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (4 chains in 2 jobs)
    NUTS: [sigma, beta, alpha]
    Sampling 4 chains: 100%|██████████| 8000/8000 [00:07<00:00, 1060.31draws/s]



```python
az.plot_trace(trace)
plt.show()
```


![png](02_Base%20Models_files/02_Base%20Models_13_0.png)


We can do the same with a Student likelihood:


```python
with pm.Model() as lin_student:
    alpha = pm.Normal("alpha", mu=0, sd=10)
    beta = pm.Normal("beta", mu=0, sd=1)
    mu = alpha + beta*d["living_space_s"]
    
    sigma = pm.HalfCauchy("sigma", 5)
    nu = pm.Gamma("nu", alpha=2, beta=0.1)
    y = pm.StudentT("y", nu=nu, mu=mu, sd=sigma, observed=d[target])
    trace = pm.sample(random_seed=2412, chains=4, 
                      draws=1000, tune=1000)
    
    prior = pm.sample_prior_predictive()
```

    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (4 chains in 2 jobs)
    NUTS: [nu, sigma, beta, alpha]
    Sampling 4 chains: 100%|██████████| 8000/8000 [00:18<00:00, 439.03draws/s]



```python
az.plot_trace(trace)
plt.show()
```


![png](02_Base%20Models_files/02_Base%20Models_16_0.png)

