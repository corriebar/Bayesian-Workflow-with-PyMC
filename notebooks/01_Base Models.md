
# Base Models


```python
import sys
sys.path.append('../src/')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pymc3 as pm

import arviz as az

from utils.data_utils import destandardize_area, destandardize_price, load_data
from utils.plot_utils import draw_model_plot, set_plot_defaults
```


```python
d, _, _ = load_data()
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
    </tr>
  </tbody>
</table>
</div>




```python
d.shape
```




    (8014, 21)



## Intercept only model

$$\begin{align*}
price &\sim \text{Normal}(\mu, \sigma)\\
\mu &\sim \text{Normal}(0, 10)\\
\sigma &\sim \text{HalfCauchy}(10)
\end{align*}$$


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
    Sampling 4 chains: 100%|██████████| 12000/12000 [00:09<00:00, 1277.22draws/s]



```python
az.plot_trace(trace, coords={"chain":[0,1]})
plt.show()
```


![png](01_Base%20Models_files/01_Base%20Models_8_0.png)



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
    Sampling 4 chains: 100%|██████████| 12000/12000 [00:27<00:00, 442.08draws/s]



```python
az.plot_trace(trace)
plt.show()
```


![png](01_Base%20Models_files/01_Base%20Models_10_0.png)


## Linear base model

$$\begin{align*}
price &\sim \text{Normal}(\mu, \sigma)\\
\mu &= \alpha + \beta \;\text{living_space}\\
\alpha &\sim \text{Normal}(0, 10) \\
\beta &\sim \text{Normal}(0, 1) \\
\sigma &\sim \text{HalfCauchy}(10)
\end{align*}$$


```python
with pm.Model() as lin_normal:
    alpha = pm.Normal("alpha", mu=0, sd=10)
    beta = pm.Normal("beta", mu=0, sd=1)
    mu = alpha + beta*d["living_space_s"]
    
    sigma = pm.HalfCauchy("sigma", 5)
    y = pm.Normal("y", mu=mu, sd=sigma, observed=d[target])
    trace = pm.sample(random_seed=2412, chains=4, 
                      draws=1000, tune=1000)
```

    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (4 chains in 2 jobs)
    NUTS: [sigma, beta, alpha]
    Sampling 4 chains: 100%|██████████| 12000/12000 [00:10<00:00, 1120.91draws/s]



```python
az.plot_trace(trace)
plt.show()
```


![png](01_Base%20Models_files/01_Base%20Models_14_0.png)



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
    Sampling 4 chains: 100%|██████████| 12000/12000 [00:26<00:00, 450.11draws/s]



```python
az.plot_trace(trace)
plt.show()
```


![png](01_Base%20Models_files/01_Base%20Models_16_0.png)


# Prior Predictive Checks

## Flat priors
Idea: have uninformative priors and thus be more objective/less subjective


```python
with pm.Model() as flat_prior:
    alpha = pm.Normal("alpha", mu=0, sd=1000)
    beta = pm.Normal("beta", mu=0, sd=1000)
    mu = alpha + beta*d["living_space_s"]
    
    sigma = pm.Exponential("sigma", 1/5)
    y = pm.Normal("y", mu=mu, sd=sigma, observed=d[target])
    
    flat_prior = pm.sample_prior_predictive()
```


```python
with pm.Model() as less_flat_prior:
    alpha = pm.Normal("alpha", mu=0, sd=100)
    beta = pm.Normal("beta", mu=0, sd=100)
    mu = alpha + beta*d["living_space_s"]
    
    sigma = pm.Exponential("sigma", 1/5)
    y = pm.Normal("y", mu=mu, sd=sigma, observed=d[target])
    
    less_flat_prior = pm.sample_prior_predictive()
```


```python
flat_data = az.from_pymc3( prior = flat_prior )
less_flat_data = az.from_pymc3( prior = less_flat_prior )
```


```python
ax = az.plot_density([flat_data, less_flat_data], group="prior",
                data_labels = ["flat", "less flat"],
               var_names = ["alpha", "beta", "sigma"],
               shade=0.3, bw=8, figsize=(20,6), credible_interval=1)
ax[0].legend(frameon=False, prop={'size': 16}, markerscale=3., loc="upper left")
ax[0].set_title("alpha", fontdict={'fontsize': 20})
ax[1].set_title("beta", fontdict={'fontsize': 20})
ax[2].set_title("sigma", fontdict={'fontsize': 20})
plt.show()
```


![png](01_Base%20Models_files/01_Base%20Models_21_0.png)



```python
area_s = np.linspace(start=-1.5, stop=3, num=50)
```


```python
fig, ax = draw_model_plot(flat_prior, title="Flat Prior")
plt.show()
```


![png](01_Base%20Models_files/01_Base%20Models_23_0.png)



```python
fig, ax = draw_model_plot(flat_prior, title="Flat Prior")
ax.axhline(y=8.5e6, c="#537d91", label="Most expensive flat sold in Berlin", lw=3)
ax.axhline(y=0, c="#f77754", label="0 €", lw=3)
ax.legend(frameon=False)
plt.show()
```


![png](01_Base%20Models_files/01_Base%20Models_24_0.png)


## Weakly informative prior

On recommendations what prior is recommended for the different parameters including what are good default parameter values, check [Prior recommendation by Stan team](https://github.com/stan-dev/stan/wiki/Prior-Choice-Recommendations)


```python
with pm.Model() as weakly_inf_prior:
    alpha = pm.Normal("alpha", mu=0, sd=20)
    beta = pm.Normal("beta", mu=0, sd=5)
    mu = alpha + beta*d["living_space_s"]
    
    sigma = pm.Exponential("sigma", lam = 1/2.5)
    y = pm.Normal("y", mu=mu, sd=sigma, observed=d[target])
    
    weakly_inf_prior = pm.sample_prior_predictive()
```


```python
weakly_inf_data = az.from_pymc3(prior=weakly_inf_prior)
```


```python
ax = az.plot_density(weakly_inf_data, group="prior",
                data_labels = ["weakly informative prior"],
               var_names = ["alpha", "beta"],
               shade=0.3, bw=8, figsize=(13,6), credible_interval=1)
ax[0].set_title("alpha", fontdict={'fontsize': 20})
ax[1].set_title("beta", fontdict={'fontsize': 20})
plt.show()
```


![png](01_Base%20Models_files/01_Base%20Models_28_0.png)



```python
axes = az.plot_density([weakly_inf_data, less_flat_data], group="prior",
                data_labels = ["weakly\ninformative", "flat"],
               var_names = ["alpha", "beta"],
               shade=0.3, bw=8, figsize=(13,6), credible_interval=1)
axes[0].set_xlim(-50, 50)
axes[1].set_xlim(-15, 15)
axes[0].legend(frameon=False, prop={'size': 16}, markerscale=3., loc="upper left")
axes[0].set_title("alpha", fontdict={'fontsize': 20})
axes[1].set_title("beta", fontdict={'fontsize': 20})
plt.show()
```


![png](01_Base%20Models_files/01_Base%20Models_29_0.png)



```python
fig, ax = draw_model_plot(weakly_inf_prior, title="Weakly Informative Prior")
plt.show()
```


![png](01_Base%20Models_files/01_Base%20Models_30_0.png)



```python
fig, ax = draw_model_plot(weakly_inf_prior, title="Weakly Informative Prior")
ax.axhline(y=8.5e6, c="#537d91", label="Most expensive flat sold in Berlin", lw=3)
ax.axhline(y=0, c="#f77754", label="0 €", lw=3)
ax.legend(frameon=False)
plt.show()
```


![png](01_Base%20Models_files/01_Base%20Models_31_0.png)



```python
with pm.Model() as inf_prior:
    alpha = pm.Normal("alpha", mu=3, sd=2.5)
    beta = pm.Normal("beta", mu=1, sd=2.5)
    mu = alpha + beta*d["living_space_s"]
    
    sigma = pm.Exponential("sigma", lam = 1/2.5)
    y = pm.Normal("y", mu=mu, sd=sigma, observed=d[target])
    
    inf_prior = pm.sample_prior_predictive()
```


```python
inf_data = az.from_pymc3(prior=inf_prior)
```


```python
az.plot_density(inf_data, group="prior",
                data_labels = ["informative prior"],
               var_names = ["alpha", "beta"],
               shade=0.3, bw=8, figsize=(13,6), credible_interval=1)
plt.show()
```


![png](01_Base%20Models_files/01_Base%20Models_34_0.png)



```python
az.plot_density([inf_data, weakly_inf_data], group="prior",
                data_labels = ["informative prior", "weakly informative prior"],
               var_names = ["alpha", "beta"],
               shade=0.3, bw=8, figsize=(13,6), credible_interval=1)
plt.show()
```


![png](01_Base%20Models_files/01_Base%20Models_35_0.png)



```python
fig, ax = draw_model_plot(inf_prior, title="Informative Prior")
ax.axhline(y=0, c="#f77754", label="0 €", lw=3)
ax.legend(frameon=False)
plt.show()
```


![png](01_Base%20Models_files/01_Base%20Models_36_0.png)



```python

```
