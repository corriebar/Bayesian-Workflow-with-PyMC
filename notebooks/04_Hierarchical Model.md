# Hierarchical Model
In this notebook, we implement the following hierarchical model in PyMC:

$$\begin{align*}
y &\sim \text{Normal}(\mu, \sigma)\\
\mu &= \alpha_{[ZIP]} + \beta_{[ZIP]} \text{area}\\
\alpha_{[ZIP]} &\sim \text{Normal}(\mu_{\alpha},\sigma_{\alpha}) \\
\beta_{[ZIP]} &\sim \text{Normal}(\mu_{\beta},\sigma_{\beta}) \\
\mu_{\alpha}, \mu_{\beta} &\sim \text{Normal}(0,100) \\
\sigma, \sigma_{\alpha}, \sigma_{\beta} &\sim \text{Exponential}(100)
\end{align*}$$

There are different ways to encode this model. The naive version is to use a center parametrization, however, for many problems this is not an optimal and a reparameterization, the non-centered version is a better alternative.
Another way to encode a hierarchical model, is to use instead a multivariate normal. I will present all three versions here.


```python
import sys
sys.path.append('../src/')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm
import theano.tensor as tt
import theano

import arviz as az

from utils.data_utils import destandardize_area, destandardize_price, load_data
from utils.plot_utils import draw_model_plot, set_plot_defaults
```

    WARNING (theano.tensor.blas): Using NumPy C-API based implementation for BLAS functions.
    /home/corrie/.pyenv/versions/anaconda3-2019.03/lib/python3.7/site-packages/xarray/core/merge.py:17: FutureWarning: The Panel class is removed from pandas. Accessing it from the top-level namespace will also be removed in the next version
      PANDAS_TYPES = (pd.Series, pd.DataFrame, pd.Panel)



```python
set_plot_defaults(font="Europace Sans")
d, zip_lookup, num_zip_codes = load_data()
zip_codes = np.sort(d.zip.unique())

target = "price_s"
```

## Not very good Model
here, I actually accidently let $\sigma$ also vary by ZIP code. The idea in itself is actually not that stupid (makes sense that some heterogeneous ZIP codes vary maybe more than others in the price) but this implementation fails, as you can see in the trace plots. I leave it here just to give an example of some bad trace plots.


```python
with pm.Model() as hier_model_naiv:
    mu_alpha = pm.Normal("mu_alpha", mu=0, sd=20)
    sigma_alpha = pm.Exponential("sigma_alpha", lam=1/2.5)
    
    mu_beta = pm.Normal("mu_beta", mu=0, sd=5)
    sigma_beta = pm.Exponential("sigma_beta", lam=1/2.5)
    
    alpha = pm.Normal("alpha", mu=mu_alpha, sd=sigma_alpha, shape=num_zip_codes)
    beta = pm.Normal("beta", mu=mu_beta, sd=sigma_beta, shape=num_zip_codes)
    # sigma also varies by zip code
    sigma = pm.Exponential("sigma", lam=1/2.5, shape=num_zip_codes)
    
    mu = alpha[d.zip_code.values] + beta[d.zip_code.values]*d.living_space_s
    y = pm.Normal("y", mu=mu, sd=sigma[d.zip_code.values], observed=d[target])
    
    prior = pm.sample_prior_predictive()
    trace = pm.sample(random_seed=2412, chains=4, 
                      draws=1000, tune=1000)
    posterior_predictive = pm.sample_posterior_predictive(trace)
```

    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (4 chains in 2 jobs)
    NUTS: [sigma, beta, alpha, sigma_beta, mu_beta, sigma_alpha, mu_alpha]
    Sampling 4 chains: 100%|██████████| 8000/8000 [10:49<00:00, 12.32draws/s] 
    There were 321 divergences after tuning. Increase `target_accept` or reparameterize.
    The acceptance probability does not match the target. It is 0.7095635801521762, but should be close to 0.8. Try to increase the number of tuning steps.
    There were 324 divergences after tuning. Increase `target_accept` or reparameterize.
    There were 389 divergences after tuning. Increase `target_accept` or reparameterize.
    The acceptance probability does not match the target. It is 0.638361360052169, but should be close to 0.8. Try to increase the number of tuning steps.
    There were 421 divergences after tuning. Increase `target_accept` or reparameterize.
    The acceptance probability does not match the target. It is 0.598018163979549, but should be close to 0.8. Try to increase the number of tuning steps.
    The gelman-rubin statistic is larger than 1.05 for some parameters. This indicates slight problems during sampling.
    The estimated number of effective samples is smaller than 200 for some parameters.
    100%|██████████| 4000/4000 [00:10<00:00, 370.32it/s]



```python
bad_model = az.from_pymc3(trace=trace,
                     prior=prior,
                     posterior_predictive=posterior_predictive,
                     coords={'zip_code': zip_codes},
                     dims={"alpha": ["zip_code"], "beta": ["zip_code"],
                          "sigma": ["zip_code"]})
bad_model
```




    Inference data with groups:
    	> posterior
    	> sample_stats
    	> posterior_predictive
    	> prior
    	> observed_data




```python
bad_model.to_netcdf("../models/bad_model.nc")
```




    '../models/bad_model.nc'




```python
az.summary(bad_model)
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
      <th>mean</th>
      <th>sd</th>
      <th>hpd_3%</th>
      <th>hpd_97%</th>
      <th>mcse_mean</th>
      <th>mcse_sd</th>
      <th>ess_mean</th>
      <th>ess_sd</th>
      <th>ess_bulk</th>
      <th>ess_tail</th>
      <th>r_hat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>mu_alpha</td>
      <td>3.601</td>
      <td>0.056</td>
      <td>3.494</td>
      <td>3.703</td>
      <td>0.003</td>
      <td>0.002</td>
      <td>293.0</td>
      <td>291.0</td>
      <td>307.0</td>
      <td>201.0</td>
      <td>1.01</td>
    </tr>
    <tr>
      <td>mu_beta</td>
      <td>2.691</td>
      <td>0.089</td>
      <td>2.534</td>
      <td>2.858</td>
      <td>0.006</td>
      <td>0.004</td>
      <td>235.0</td>
      <td>235.0</td>
      <td>239.0</td>
      <td>49.0</td>
      <td>1.02</td>
    </tr>
    <tr>
      <td>alpha[0]</td>
      <td>4.989</td>
      <td>0.161</td>
      <td>4.661</td>
      <td>5.244</td>
      <td>0.009</td>
      <td>0.006</td>
      <td>326.0</td>
      <td>318.0</td>
      <td>345.0</td>
      <td>1456.0</td>
      <td>1.02</td>
    </tr>
    <tr>
      <td>alpha[1]</td>
      <td>5.121</td>
      <td>0.273</td>
      <td>4.615</td>
      <td>5.614</td>
      <td>0.016</td>
      <td>0.011</td>
      <td>287.0</td>
      <td>287.0</td>
      <td>291.0</td>
      <td>2053.0</td>
      <td>1.02</td>
    </tr>
    <tr>
      <td>alpha[2]</td>
      <td>5.040</td>
      <td>0.258</td>
      <td>4.596</td>
      <td>5.541</td>
      <td>0.018</td>
      <td>0.013</td>
      <td>212.0</td>
      <td>212.0</td>
      <td>226.0</td>
      <td>435.0</td>
      <td>1.02</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>sigma[214]</td>
      <td>1.808</td>
      <td>1.719</td>
      <td>0.015</td>
      <td>4.724</td>
      <td>0.056</td>
      <td>0.040</td>
      <td>944.0</td>
      <td>944.0</td>
      <td>682.0</td>
      <td>1119.0</td>
      <td>1.01</td>
    </tr>
    <tr>
      <td>sigma[215]</td>
      <td>0.903</td>
      <td>0.953</td>
      <td>0.011</td>
      <td>2.433</td>
      <td>0.038</td>
      <td>0.027</td>
      <td>644.0</td>
      <td>644.0</td>
      <td>482.0</td>
      <td>940.0</td>
      <td>1.01</td>
    </tr>
    <tr>
      <td>sigma[216]</td>
      <td>1.953</td>
      <td>1.718</td>
      <td>0.029</td>
      <td>5.444</td>
      <td>0.202</td>
      <td>0.144</td>
      <td>72.0</td>
      <td>72.0</td>
      <td>110.0</td>
      <td>31.0</td>
      <td>1.02</td>
    </tr>
    <tr>
      <td>sigma[217]</td>
      <td>1.838</td>
      <td>1.750</td>
      <td>0.013</td>
      <td>4.870</td>
      <td>0.084</td>
      <td>0.060</td>
      <td>430.0</td>
      <td>430.0</td>
      <td>172.0</td>
      <td>128.0</td>
      <td>1.02</td>
    </tr>
    <tr>
      <td>sigma[218]</td>
      <td>3.763</td>
      <td>2.413</td>
      <td>0.032</td>
      <td>7.882</td>
      <td>0.099</td>
      <td>0.070</td>
      <td>593.0</td>
      <td>593.0</td>
      <td>494.0</td>
      <td>516.0</td>
      <td>1.02</td>
    </tr>
  </tbody>
</table>
<p>661 rows × 11 columns</p>
</div>




```python
az.plot_trace(bad_model, var_names=["mu_alpha", "mu_beta", "sigma_alpha", "sigma_beta"])
plt.show()
```


![png](04_Hierarchical%20Model_files/04_Hierarchical%20Model_8_0.png)


## Centered Parameterization
The centered parameterization uses the same parameter description as in the model description above. For many problems, this is not the recommended parametrization, as discussed [here](https://twiecki.io/blog/2017/02/08/bayesian-hierchical-non-centered/) by Thomas Wiecki, one of the PyMC core devs.

However, in some cases, the centered parametrization actually performs better than the non-centered parameterization, see e.g. this [discussion](https://discourse.mc-stan.org/t/centered-vs-non-centered-parameterizations/7344).


```python
zips = theano.shared(d["zip_code"].values)
# idx variables cannnot used with pm.Data() so far, because of bug
# see here: https://discourse.pymc.io/t/integer-values-with-pm-data/3776
# and here: https://github.com/pymc-devs/pymc3/issues/3493

with pm.Model() as centered_hier_model:
    area = pm.Data("area", d["living_space_s"])
    #zips = pm.Data("zips", d["zip_code"].values)
    
    mu_alpha = pm.Normal("mu_alpha", mu=0, sd=20)
    sigma_alpha = pm.Exponential("sigma_alpha", lam=1/5)
    
    mu_beta = pm.Normal("mu_beta", mu=0, sd=5)
    sigma_beta = pm.Exponential("sigma_beta", lam=1/5)
    
    alpha = pm.Normal("alpha", mu=mu_alpha, sd=sigma_alpha, shape=num_zip_codes)
    beta = pm.Normal("beta", mu=mu_beta, sd=sigma_beta, shape=num_zip_codes)
    # without varying sigma
    sigma = pm.Exponential("sigma", lam=1/5)
    
    mu = alpha[zips] + beta[zips]*area
    y = pm.Normal("y", mu=mu, sd=sigma, observed=d[target])
    
    prior = pm.sample_prior_predictive()
    trace = pm.sample(random_seed=2412, chains=4, 
                      draws=1000, tune=1000)
    posterior_predictive = pm.sample_posterior_predictive(trace)
```

    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (4 chains in 2 jobs)
    NUTS: [sigma, beta, alpha, sigma_beta, mu_beta, sigma_alpha, mu_alpha]
    Sampling 4 chains: 100%|██████████| 8000/8000 [01:50<00:00, 72.36draws/s] 
    100%|██████████| 4000/4000 [00:09<00:00, 409.70it/s]



```python
data = az.from_pymc3(trace=trace,
                     prior=prior,
                     posterior_predictive=posterior_predictive,
                     coords={'zip_code': zip_codes},
                     dims={"alpha": ["zip_code"], "beta": ["zip_code"]})
data
```




    Inference data with groups:
    	> posterior
    	> sample_stats
    	> posterior_predictive
    	> prior
    	> observed_data
    	> constant_data




```python
pm.save_trace(trace, directory="../models/centered_hier")
```


```python
data.to_netcdf("../models/centered_hier.nc")
```




    '../models/centered_hier.nc'




```python
az.plot_trace(data.posterior.where(data.posterior.zip_code.isin(["12047"]), drop=True),
             var_names=["alpha"])
plt.show()
```


![png](04_Hierarchical%20Model_files/04_Hierarchical%20Model_14_0.png)



```python
az.summary(trace)
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
      <th>mean</th>
      <th>sd</th>
      <th>hpd_3%</th>
      <th>hpd_97%</th>
      <th>mcse_mean</th>
      <th>mcse_sd</th>
      <th>ess_mean</th>
      <th>ess_sd</th>
      <th>ess_bulk</th>
      <th>ess_tail</th>
      <th>r_hat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>mu_alpha</td>
      <td>3.642</td>
      <td>0.057</td>
      <td>3.527</td>
      <td>3.741</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>5524.0</td>
      <td>5524.0</td>
      <td>5535.0</td>
      <td>3184.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>mu_beta</td>
      <td>2.775</td>
      <td>0.099</td>
      <td>2.596</td>
      <td>2.966</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>5782.0</td>
      <td>5781.0</td>
      <td>5766.0</td>
      <td>3343.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>alpha[0]</td>
      <td>4.976</td>
      <td>0.161</td>
      <td>4.688</td>
      <td>5.289</td>
      <td>0.002</td>
      <td>0.002</td>
      <td>5618.0</td>
      <td>5590.0</td>
      <td>5626.0</td>
      <td>3249.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>alpha[1]</td>
      <td>5.097</td>
      <td>0.289</td>
      <td>4.545</td>
      <td>5.616</td>
      <td>0.004</td>
      <td>0.003</td>
      <td>6018.0</td>
      <td>6018.0</td>
      <td>6017.0</td>
      <td>3309.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>alpha[2]</td>
      <td>5.128</td>
      <td>0.201</td>
      <td>4.731</td>
      <td>5.483</td>
      <td>0.002</td>
      <td>0.002</td>
      <td>7223.0</td>
      <td>7223.0</td>
      <td>7272.0</td>
      <td>3517.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>beta[217]</td>
      <td>2.668</td>
      <td>1.280</td>
      <td>0.290</td>
      <td>4.958</td>
      <td>0.014</td>
      <td>0.012</td>
      <td>7809.0</td>
      <td>5449.0</td>
      <td>7833.0</td>
      <td>2742.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>beta[218]</td>
      <td>0.504</td>
      <td>0.792</td>
      <td>-0.999</td>
      <td>1.999</td>
      <td>0.009</td>
      <td>0.011</td>
      <td>7653.0</td>
      <td>2519.0</td>
      <td>7656.0</td>
      <td>3127.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>sigma_alpha</td>
      <td>0.741</td>
      <td>0.044</td>
      <td>0.664</td>
      <td>0.830</td>
      <td>0.001</td>
      <td>0.000</td>
      <td>4491.0</td>
      <td>4491.0</td>
      <td>4466.0</td>
      <td>3015.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>sigma_beta</td>
      <td>1.274</td>
      <td>0.071</td>
      <td>1.146</td>
      <td>1.415</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>4779.0</td>
      <td>4779.0</td>
      <td>4726.0</td>
      <td>3035.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>sigma</td>
      <td>1.203</td>
      <td>0.010</td>
      <td>1.184</td>
      <td>1.221</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>7403.0</td>
      <td>7403.0</td>
      <td>7381.0</td>
      <td>3018.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>443 rows × 11 columns</p>
</div>




```python
az.plot_trace(trace, var_names=["mu_alpha", "mu_beta", "sigma_alpha", "sigma_beta", "sigma"])
plt.show()
```


![png](04_Hierarchical%20Model_files/04_Hierarchical%20Model_16_0.png)


## Non-Centered Version
Often, this reparametrization works better. For a more detailed explanation, I refer again to the [blog post](https://twiecki.io/blog/2017/02/08/bayesian-hierchical-non-centered/) by Thomas Wiecki.


```python
with pm.Model() as non_centered_hier:
    
    beta_offset = pm.Normal("beta_offset", mu=0, sd=1, shape=num_zip_codes)
    
    mu_beta = pm.Normal("mu_beta", mu=0, sd=5)
    sigma_beta = pm.Exponential("sigma_beta", 1/5)
    beta = mu_beta + beta_offset*sigma_beta
    
    alpha_offset = pm.Normal("alpha_offset", mu=0, sd=1, shape=num_zip_codes)
    
    mu_alpha = pm.Normal("mu_alpha", mu=0, sd=20)
    sigma_alpha = pm.Exponential("sigma_alpha", 1/5)
    alpha = mu_alpha + alpha_offset*sigma_alpha
    
    mu = alpha[d.zip_code.values] + beta[d.zip_code.values]*d.living_space_s
    sigma = pm.Exponential("sigma", 1/5)
    y = pm.Normal("y", mu=mu, sd=sigma, observed=d[target])
    
    prior = pm.sample_prior_predictive()
    trace = pm.sample(random_seed=2412, chains=4, 
                      draws=1000, tune=1000)
    posterior_predictive = pm.sample_posterior_predictive(trace)
```

    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (4 chains in 2 jobs)
    NUTS: [sigma, sigma_alpha, mu_alpha, alpha_offset, sigma_beta, mu_beta, beta_offset]
    Sampling 4 chains: 100%|██████████| 8000/8000 [03:29<00:00, 38.28draws/s]
    The number of effective samples is smaller than 25% for some parameters.
    100%|██████████| 4000/4000 [00:10<00:00, 364.94it/s]



```python
az.summary(trace)
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
      <th>mean</th>
      <th>sd</th>
      <th>hpd_3%</th>
      <th>hpd_97%</th>
      <th>mcse_mean</th>
      <th>mcse_sd</th>
      <th>ess_mean</th>
      <th>ess_sd</th>
      <th>ess_bulk</th>
      <th>ess_tail</th>
      <th>r_hat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>beta_offset[0]</td>
      <td>0.737</td>
      <td>0.250</td>
      <td>0.279</td>
      <td>1.213</td>
      <td>0.004</td>
      <td>0.003</td>
      <td>3385.0</td>
      <td>2935.0</td>
      <td>3383.0</td>
      <td>2582.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <td>beta_offset[1]</td>
      <td>0.868</td>
      <td>0.455</td>
      <td>0.013</td>
      <td>1.702</td>
      <td>0.006</td>
      <td>0.005</td>
      <td>6135.0</td>
      <td>4083.0</td>
      <td>6237.0</td>
      <td>2802.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <td>beta_offset[2]</td>
      <td>1.293</td>
      <td>0.323</td>
      <td>0.686</td>
      <td>1.913</td>
      <td>0.005</td>
      <td>0.003</td>
      <td>4578.0</td>
      <td>4578.0</td>
      <td>4561.0</td>
      <td>3079.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <td>beta_offset[3]</td>
      <td>-0.205</td>
      <td>0.938</td>
      <td>-1.984</td>
      <td>1.525</td>
      <td>0.010</td>
      <td>0.016</td>
      <td>9016.0</td>
      <td>1708.0</td>
      <td>9001.0</td>
      <td>3022.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <td>beta_offset[4]</td>
      <td>-0.079</td>
      <td>0.949</td>
      <td>-1.994</td>
      <td>1.626</td>
      <td>0.009</td>
      <td>0.016</td>
      <td>10836.0</td>
      <td>1722.0</td>
      <td>10794.0</td>
      <td>2975.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>alpha_offset[218]</td>
      <td>-0.728</td>
      <td>0.940</td>
      <td>-2.407</td>
      <td>1.085</td>
      <td>0.011</td>
      <td>0.011</td>
      <td>7589.0</td>
      <td>3522.0</td>
      <td>7606.0</td>
      <td>3218.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <td>mu_alpha</td>
      <td>3.640</td>
      <td>0.054</td>
      <td>3.539</td>
      <td>3.739</td>
      <td>0.002</td>
      <td>0.001</td>
      <td>846.0</td>
      <td>846.0</td>
      <td>846.0</td>
      <td>1481.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <td>sigma_beta</td>
      <td>1.274</td>
      <td>0.073</td>
      <td>1.141</td>
      <td>1.412</td>
      <td>0.003</td>
      <td>0.002</td>
      <td>688.0</td>
      <td>687.0</td>
      <td>688.0</td>
      <td>1545.0</td>
      <td>1.01</td>
    </tr>
    <tr>
      <td>sigma_alpha</td>
      <td>0.741</td>
      <td>0.043</td>
      <td>0.660</td>
      <td>0.819</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>1193.0</td>
      <td>1193.0</td>
      <td>1189.0</td>
      <td>2030.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <td>sigma</td>
      <td>1.203</td>
      <td>0.010</td>
      <td>1.184</td>
      <td>1.222</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>9312.0</td>
      <td>9312.0</td>
      <td>9267.0</td>
      <td>2736.0</td>
      <td>1.01</td>
    </tr>
  </tbody>
</table>
<p>443 rows × 11 columns</p>
</div>




```python
az.plot_trace(trace, var_names=["mu_alpha", "mu_beta", "sigma_alpha", "sigma_beta", "sigma"])
plt.show()
```


![png](04_Hierarchical%20Model_files/04_Hierarchical%20Model_20_0.png)


## Multivariate Model (includes correlation)
Another way of coding a hierarchical model is to use a multivariate normal distribution for the random effects:
$$\begin{align*}
y &\sim \text{Normal}(\mu, \sigma)\\
\mu &= \alpha_{[ZIP]} + \beta_{[ZIP]} \text{area}\\
\\
\begin{bmatrix}\alpha_{[ZIP]} \\ \beta_{[ZIP]} \end{bmatrix} &\sim \text{Normal}( \begin{bmatrix} \mu_{\alpha} \\ \mu_{\beta} \end{bmatrix}, \Sigma)\\
\mu_{\alpha}, \mu_{\beta} &\sim \text{Normal}(0,100) \\
\end{align*}$$

Setting the prior for the $\Sigma$ matrix is a bit more involved. The [Stan User Guide](https://mc-stan.org/docs/2_18/stan-users-guide/multivariate-hierarchical-priors-section.html) gives some explanation for how to pick the matrix prior.


```python
with pm.Model() as correlation_hier_model: 
    # model specification is adapted from 
    # https://stackoverflow.com/questions/39364919/pymc3-how-to-model-correlated-intercept-and-slope-in-multilevel-linear-regressi
    # mu is mean vector for alpha and beta parameter
    mu = pm.Normal("mu", mu=0, sd=5, shape=(2,)) 
    # sigma is scale vector for alpha and beta parameter
    sigma = pm.Exponential("sigma", 1/5, shape=(2,))
    
    C_triu = pm.LKJCorr("C_triu", n=2, p=2)
    C = tt.fill_diagonal(C_triu[np.zeros((2,2), 'int')], 1.)
    # scale vector becomes diagonal of covariance matrix
    sigma_diag = tt.nlinalg.diag(sigma)
    cov = tt.nlinalg.matrix_dot(sigma_diag, C, sigma_diag)
    tau = tt.nlinalg.matrix_inverse(cov)
    
    # alpha and beta come from a multivariate normal
    # with mean vector mu
    # and covariance matrix tau 
    # tau includes scale for alpha and beta on the diagonals
    # and the correlation on  between alpha and beta off the diagonals
    ab = pm.MvNormal("ab", mu=mu, tau=tau, shape=(num_zip_codes, 2))
    
    sd_price = pm.Exponential("sd_price", lam=1/5)
    
    # compute estimate for price
    mean_price = ab[:,0][d.zip_code.values] + ab[:,1][d.zip_code.values]*d.living_space_s
    y = pm.Normal("y", mu=mean_price, sd=sd_price, observed=d[target])
    
    #prior = pm.sample_prior_predictive()
    trace = pm.sample(random_seed=2412, chains=4, 
                      draws=1000, tune=1000)
    posterior_predictive = pm.sample_posterior_predictive(trace)
```

    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    /home/corrie/.pyenv/versions/anaconda3-2019.03/lib/python3.7/site-packages/theano/tensor/basic.py:6611: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      result[diagonal_slice] = x
    /home/corrie/.pyenv/versions/anaconda3-2019.03/lib/python3.7/site-packages/theano/tensor/basic.py:6611: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      result[diagonal_slice] = x
    /home/corrie/.pyenv/versions/anaconda3-2019.03/lib/python3.7/site-packages/theano/tensor/basic.py:6611: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      result[diagonal_slice] = x
    /home/corrie/.pyenv/versions/anaconda3-2019.03/lib/python3.7/site-packages/theano/tensor/basic.py:6611: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      result[diagonal_slice] = x
    /home/corrie/.pyenv/versions/anaconda3-2019.03/lib/python3.7/site-packages/theano/tensor/basic.py:6611: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      result[diagonal_slice] = x
    /home/corrie/.pyenv/versions/anaconda3-2019.03/lib/python3.7/site-packages/theano/tensor/basic.py:6611: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      result[diagonal_slice] = x
    Multiprocess sampling (4 chains in 2 jobs)
    NUTS: [sd_price, ab, C_triu, sigma, mu]
    Sampling 4 chains:   0%|          | 0/8000 [00:00<?, ?draws/s]/home/corrie/.pyenv/versions/anaconda3-2019.03/lib/python3.7/site-packages/theano/tensor/basic.py:6611: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      result[diagonal_slice] = x
    /home/corrie/.pyenv/versions/anaconda3-2019.03/lib/python3.7/site-packages/theano/tensor/basic.py:6611: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      result[diagonal_slice] = x
    Sampling 4 chains:  50%|████▉     | 3965/8000 [05:31<02:33, 26.33draws/s] /home/corrie/.pyenv/versions/anaconda3-2019.03/lib/python3.7/site-packages/theano/tensor/basic.py:6611: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      result[diagonal_slice] = x
    Sampling 4 chains:  50%|█████     | 4004/8000 [05:34<05:52, 11.34draws/s]/home/corrie/.pyenv/versions/anaconda3-2019.03/lib/python3.7/site-packages/theano/tensor/basic.py:6611: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      result[diagonal_slice] = x
    Sampling 4 chains: 100%|██████████| 8000/8000 [10:43<00:00, 12.43draws/s]
    100%|██████████| 4000/4000 [00:09<00:00, 406.16it/s]



```python
mvn_data = az.from_pymc3(trace=trace,
                     prior=prior,
                     posterior_predictive=posterior_predictive,
                     coords={'zip_code': zip_codes, 'param': ["alpha", "beta"],
                            "sigma_dim_0": ["alpha", "beta"]},
                     dims={"ab": ["zip_code", "param"],
                          "mu": ["param"]})
mvn_data
```




    Inference data with groups:
    	> posterior
    	> sample_stats
    	> posterior_predictive
    	> prior
    	> observed_data




```python
mvn_data.to_netcdf("../models/mvn_hier.nc")
```




    '../models/mvn_hier.nc'




```python
pm.save_trace(trace, directory="../models/mvn_hier")
```


```python
az.summary(mvn_data)
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
      <th>mean</th>
      <th>sd</th>
      <th>hpd_3%</th>
      <th>hpd_97%</th>
      <th>mcse_mean</th>
      <th>mcse_sd</th>
      <th>ess_mean</th>
      <th>ess_sd</th>
      <th>ess_bulk</th>
      <th>ess_tail</th>
      <th>r_hat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>mu[0]</td>
      <td>3.648</td>
      <td>0.059</td>
      <td>3.534</td>
      <td>3.754</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>2995.0</td>
      <td>2995.0</td>
      <td>3018.0</td>
      <td>2698.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>mu[1]</td>
      <td>2.762</td>
      <td>0.102</td>
      <td>2.578</td>
      <td>2.955</td>
      <td>0.002</td>
      <td>0.001</td>
      <td>2870.0</td>
      <td>2869.0</td>
      <td>2877.0</td>
      <td>3188.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>ab[0,0]</td>
      <td>4.992</td>
      <td>0.168</td>
      <td>4.686</td>
      <td>5.321</td>
      <td>0.002</td>
      <td>0.002</td>
      <td>4780.0</td>
      <td>4780.0</td>
      <td>4766.0</td>
      <td>2622.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>ab[0,1]</td>
      <td>3.817</td>
      <td>0.310</td>
      <td>3.240</td>
      <td>4.404</td>
      <td>0.004</td>
      <td>0.003</td>
      <td>4836.0</td>
      <td>4836.0</td>
      <td>4840.0</td>
      <td>3241.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>ab[1,0]</td>
      <td>5.213</td>
      <td>0.299</td>
      <td>4.694</td>
      <td>5.808</td>
      <td>0.005</td>
      <td>0.004</td>
      <td>3573.0</td>
      <td>3573.0</td>
      <td>3573.0</td>
      <td>3478.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>ab[218,1]</td>
      <td>0.618</td>
      <td>0.629</td>
      <td>-0.532</td>
      <td>1.818</td>
      <td>0.008</td>
      <td>0.007</td>
      <td>6731.0</td>
      <td>3561.0</td>
      <td>6723.0</td>
      <td>3093.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>sigma[0]</td>
      <td>0.768</td>
      <td>0.044</td>
      <td>0.685</td>
      <td>0.852</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>2338.0</td>
      <td>2328.0</td>
      <td>2357.0</td>
      <td>2385.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>sigma[1]</td>
      <td>1.293</td>
      <td>0.073</td>
      <td>1.159</td>
      <td>1.430</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>2648.0</td>
      <td>2646.0</td>
      <td>2655.0</td>
      <td>3208.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>C_triu[0]</td>
      <td>0.716</td>
      <td>0.044</td>
      <td>0.633</td>
      <td>0.795</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>1920.0</td>
      <td>1887.0</td>
      <td>1859.0</td>
      <td>2468.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>sd_price</td>
      <td>1.203</td>
      <td>0.010</td>
      <td>1.186</td>
      <td>1.221</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>5283.0</td>
      <td>5283.0</td>
      <td>5303.0</td>
      <td>3019.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>444 rows × 11 columns</p>
</div>




```python

```
