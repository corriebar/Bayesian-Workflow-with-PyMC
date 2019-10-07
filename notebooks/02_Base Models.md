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

    /home/corrie/.pyenv/versions/anaconda3-2019.03/lib/python3.7/site-packages/xarray/core/merge.py:17: FutureWarning: The Panel class is removed from pandas. Accessing it from the top-level namespace will also be removed in the next version
      PANDAS_TYPES = (pd.Series, pd.DataFrame, pd.Panel)



```python
d, _, _ = load_data()
set_plot_defaults("Europace Sans")
target = "price_s"
```


```python
d.drop(columns=["lat", "lng"]).head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ags</th>
      <th>zip</th>
      <th>price</th>
      <th>price_s</th>
      <th>log_price</th>
      <th>log_price_s</th>
      <th>sqm_price</th>
      <th>log_sqm_price</th>
      <th>log_sqm_price_s</th>
      <th>living_space</th>
      <th>living_space_s</th>
      <th>sale_year</th>
      <th>sale_month</th>
      <th>const_year</th>
      <th>const_year_s</th>
      <th>objecttype</th>
      <th>housetype</th>
      <th>usage</th>
      <th>interior_qual</th>
      <th>zip_code</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>3</td>
      <td>11000000001000000094</td>
      <td>12489</td>
      <td>180000.0</td>
      <td>1.800</td>
      <td>12.100712</td>
      <td>-0.814839</td>
      <td>2246.069379</td>
      <td>7.716937</td>
      <td>-1.009445</td>
      <td>80.14</td>
      <td>-0.276690</td>
      <td>2018</td>
      <td>8</td>
      <td>1936.0</td>
      <td>-1.210219</td>
      <td>flat</td>
      <td>NaN</td>
      <td>rent_out</td>
      <td>normal</td>
      <td>117</td>
    </tr>
    <tr>
      <td>4</td>
      <td>11000000001000000094</td>
      <td>12489</td>
      <td>87000.0</td>
      <td>0.870</td>
      <td>11.373663</td>
      <td>-2.045434</td>
      <td>1740.000000</td>
      <td>7.461640</td>
      <td>-1.700113</td>
      <td>50.00</td>
      <td>-0.671528</td>
      <td>2017</td>
      <td>7</td>
      <td>1910.0</td>
      <td>-1.913696</td>
      <td>flat</td>
      <td>NaN</td>
      <td>rent_out</td>
      <td>NaN</td>
      <td>117</td>
    </tr>
    <tr>
      <td>6</td>
      <td>11000000001000000140</td>
      <td>12489</td>
      <td>101200.0</td>
      <td>1.012</td>
      <td>11.524854</td>
      <td>-1.789530</td>
      <td>3066.666667</td>
      <td>8.028346</td>
      <td>-0.166971</td>
      <td>33.00</td>
      <td>-0.894229</td>
      <td>2016</td>
      <td>4</td>
      <td>2016.0</td>
      <td>0.954326</td>
      <td>flat</td>
      <td>NaN</td>
      <td>rent_out</td>
      <td>NaN</td>
      <td>117</td>
    </tr>
    <tr>
      <td>9</td>
      <td>11000000001000000140</td>
      <td>12489</td>
      <td>148400.0</td>
      <td>1.484</td>
      <td>11.907667</td>
      <td>-1.141586</td>
      <td>3420.142890</td>
      <td>8.137438</td>
      <td>0.128159</td>
      <td>43.39</td>
      <td>-0.758119</td>
      <td>2017</td>
      <td>1</td>
      <td>2017.0</td>
      <td>0.981383</td>
      <td>flat</td>
      <td>NaN</td>
      <td>rent_out</td>
      <td>sophisticated</td>
      <td>117</td>
    </tr>
    <tr>
      <td>10</td>
      <td>11000000001000000140</td>
      <td>12489</td>
      <td>261200.0</td>
      <td>2.612</td>
      <td>12.473042</td>
      <td>-0.184638</td>
      <td>3374.677003</td>
      <td>8.124055</td>
      <td>0.091954</td>
      <td>77.40</td>
      <td>-0.312584</td>
      <td>2016</td>
      <td>6</td>
      <td>2016.0</td>
      <td>0.954326</td>
      <td>flat</td>
      <td>NaN</td>
      <td>rent_out</td>
      <td>sophisticated</td>
      <td>117</td>
    </tr>
  </tbody>
</table>
</div>



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
    Sampling 4 chains: 100%|██████████| 8000/8000 [00:15<00:00, 509.15draws/s]



```python
az.plot_trace(trace, coords={"chain":[0,1]})
plt.show()
```


![png](02_Base%20Models_files/02_Base%20Models_7_0.png)


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
    Sampling 4 chains: 100%|██████████| 8000/8000 [00:32<00:00, 242.69draws/s]



```python
az.plot_trace(trace)
plt.show()
```


![png](02_Base%20Models_files/02_Base%20Models_10_0.png)


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
    Sampling 4 chains: 100%|██████████| 8000/8000 [00:20<00:00, 393.26draws/s]



```python
az.plot_trace(trace)
plt.show()
```


![png](02_Base%20Models_files/02_Base%20Models_14_0.png)


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
    Sampling 4 chains: 100%|██████████| 8000/8000 [00:37<00:00, 212.33draws/s]



```python
az.plot_trace(trace)
plt.show()
```


![png](02_Base%20Models_files/02_Base%20Models_17_0.png)

