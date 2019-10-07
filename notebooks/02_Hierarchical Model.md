
# Hierarchical Model


```python
import sys
sys.path.append('../src/')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pymc3 as pm
import theano.tensor as tt

import arviz as az

from utils.data_utils import destandardize_area, destandardize_price, load_data
from utils.plot_utils import draw_model_plot, set_plot_defaults
```


```python
set_plot_defaults(font="Europace Sans")
d, zip_lookup, num_zip_codes load_data()

target = "price_s"
```

## Naive Model


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
    Sampling 4 chains: 100%|██████████| 8000/8000 [06:33<00:00, 20.32draws/s]
    There were 448 divergences after tuning. Increase `target_accept` or reparameterize.
    The acceptance probability does not match the target. It is 0.6231341310244187, but should be close to 0.8. Try to increase the number of tuning steps.
    There were 258 divergences after tuning. Increase `target_accept` or reparameterize.
    There were 323 divergences after tuning. Increase `target_accept` or reparameterize.
    The acceptance probability does not match the target. It is 0.7005937762039804, but should be close to 0.8. Try to increase the number of tuning steps.
    There were 396 divergences after tuning. Increase `target_accept` or reparameterize.
    The acceptance probability does not match the target. It is 0.648914965978378, but should be close to 0.8. Try to increase the number of tuning steps.
    The estimated number of effective samples is smaller than 200 for some parameters.
    100%|██████████| 4000/4000 [00:06<00:00, 585.31it/s]



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
      <td>3.603</td>
      <td>0.058</td>
      <td>3.490</td>
      <td>3.711</td>
      <td>0.002</td>
      <td>0.001</td>
      <td>808.0</td>
      <td>805.0</td>
      <td>821.0</td>
      <td>1142.0</td>
      <td>1.01</td>
    </tr>
    <tr>
      <td>mu_beta</td>
      <td>2.699</td>
      <td>0.090</td>
      <td>2.515</td>
      <td>2.849</td>
      <td>0.004</td>
      <td>0.003</td>
      <td>641.0</td>
      <td>641.0</td>
      <td>643.0</td>
      <td>1105.0</td>
      <td>1.01</td>
    </tr>
    <tr>
      <td>alpha[0]</td>
      <td>4.967</td>
      <td>0.164</td>
      <td>4.639</td>
      <td>5.256</td>
      <td>0.005</td>
      <td>0.004</td>
      <td>1056.0</td>
      <td>1056.0</td>
      <td>1100.0</td>
      <td>395.0</td>
      <td>1.01</td>
    </tr>
    <tr>
      <td>alpha[1]</td>
      <td>5.134</td>
      <td>0.272</td>
      <td>4.558</td>
      <td>5.591</td>
      <td>0.007</td>
      <td>0.005</td>
      <td>1482.0</td>
      <td>1482.0</td>
      <td>1481.0</td>
      <td>1360.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <td>alpha[2]</td>
      <td>5.057</td>
      <td>0.242</td>
      <td>4.628</td>
      <td>5.527</td>
      <td>0.008</td>
      <td>0.005</td>
      <td>1014.0</td>
      <td>1007.0</td>
      <td>1024.0</td>
      <td>2318.0</td>
      <td>1.01</td>
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
      <td>1.915</td>
      <td>1.809</td>
      <td>0.014</td>
      <td>5.068</td>
      <td>0.072</td>
      <td>0.051</td>
      <td>630.0</td>
      <td>630.0</td>
      <td>490.0</td>
      <td>664.0</td>
      <td>1.01</td>
    </tr>
    <tr>
      <td>sigma[215]</td>
      <td>0.870</td>
      <td>0.934</td>
      <td>0.014</td>
      <td>2.486</td>
      <td>0.037</td>
      <td>0.026</td>
      <td>644.0</td>
      <td>644.0</td>
      <td>485.0</td>
      <td>895.0</td>
      <td>1.01</td>
    </tr>
    <tr>
      <td>sigma[216]</td>
      <td>1.644</td>
      <td>1.437</td>
      <td>0.023</td>
      <td>4.254</td>
      <td>0.045</td>
      <td>0.032</td>
      <td>1039.0</td>
      <td>1039.0</td>
      <td>734.0</td>
      <td>764.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <td>sigma[217]</td>
      <td>1.816</td>
      <td>1.760</td>
      <td>0.013</td>
      <td>5.235</td>
      <td>0.157</td>
      <td>0.111</td>
      <td>126.0</td>
      <td>126.0</td>
      <td>245.0</td>
      <td>106.0</td>
      <td>1.02</td>
    </tr>
    <tr>
      <td>sigma[218]</td>
      <td>3.690</td>
      <td>2.309</td>
      <td>0.060</td>
      <td>7.673</td>
      <td>0.065</td>
      <td>0.046</td>
      <td>1255.0</td>
      <td>1255.0</td>
      <td>1020.0</td>
      <td>904.0</td>
      <td>1.00</td>
    </tr>
  </tbody>
</table>
<p>661 rows × 11 columns</p>
</div>




```python
az.plot_trace(bad_model, var_names=["mu_alpha", "mu_beta", "sigma_alpha", "sigma_beta"])
plt.show()
```


![png](02_Hierarchical%20Model_files/02_Hierarchical%20Model_8_0.png)


## Centered Parameterization
In some cases, this actually performs better than the non-centered parameterization, see this [discussion](https://discourse.mc-stan.org/t/centered-vs-non-centered-parameterizations/7344)


```python
with pm.Model() as centered_hier_model:
    mu_alpha = pm.Normal("mu_alpha", mu=0, sd=20)
    sigma_alpha = pm.Exponential("sigma_alpha", lam=1/5)
    
    mu_beta = pm.Normal("mu_beta", mu=0, sd=5)
    sigma_beta = pm.Exponential("sigma_beta", lam=1/5)
    
    alpha = pm.Normal("alpha", mu=mu_alpha, sd=sigma_alpha, shape=num_zip_codes)
    beta = pm.Normal("beta", mu=mu_beta, sd=sigma_beta, shape=num_zip_codes)
    # without varying sigma
    sigma = pm.Exponential("sigma", lam=1/5)
    
    mu = alpha[d.zip_code.values] + beta[d.zip_code.values]*d.living_space_s
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
    Sampling 4 chains: 100%|██████████| 8000/8000 [01:04<00:00, 124.92draws/s]
    100%|██████████| 4000/4000 [00:05<00:00, 715.45it/s]



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




```python
az.plot_trace(data.posterior.where(data.posterior.zip_code.isin(["12047"]), drop=True),
             var_names=["alpha"])
```




    array([[<matplotlib.axes._subplots.AxesSubplot object at 0x7f19c60d7f28>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7f19c5ffad30>]],
          dtype=object)




![png](02_Hierarchical%20Model_files/02_Hierarchical%20Model_12_1.png)



```python
data.to_netcdf("../models/centered_hier.nc")
```




    '../models/centered_hier.nc'




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
      <td>3.641</td>
      <td>0.056</td>
      <td>3.534</td>
      <td>3.741</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>5877.0</td>
      <td>5872.0</td>
      <td>5867.0</td>
      <td>3246.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>mu_beta</td>
      <td>2.774</td>
      <td>0.097</td>
      <td>2.593</td>
      <td>2.957</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>5797.0</td>
      <td>5784.0</td>
      <td>5804.0</td>
      <td>3104.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>alpha[0]</td>
      <td>4.978</td>
      <td>0.163</td>
      <td>4.671</td>
      <td>5.275</td>
      <td>0.002</td>
      <td>0.001</td>
      <td>6398.0</td>
      <td>6398.0</td>
      <td>6414.0</td>
      <td>3328.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>alpha[1]</td>
      <td>5.097</td>
      <td>0.287</td>
      <td>4.564</td>
      <td>5.644</td>
      <td>0.004</td>
      <td>0.003</td>
      <td>5885.0</td>
      <td>5860.0</td>
      <td>5865.0</td>
      <td>3322.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>alpha[2]</td>
      <td>5.131</td>
      <td>0.199</td>
      <td>4.768</td>
      <td>5.506</td>
      <td>0.002</td>
      <td>0.002</td>
      <td>6460.0</td>
      <td>6460.0</td>
      <td>6471.0</td>
      <td>3174.0</td>
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
      <td>2.667</td>
      <td>1.292</td>
      <td>0.292</td>
      <td>5.160</td>
      <td>0.014</td>
      <td>0.012</td>
      <td>8101.0</td>
      <td>5462.0</td>
      <td>8127.0</td>
      <td>2891.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>beta[218]</td>
      <td>0.493</td>
      <td>0.794</td>
      <td>-1.043</td>
      <td>1.890</td>
      <td>0.010</td>
      <td>0.011</td>
      <td>6097.0</td>
      <td>2854.0</td>
      <td>6089.0</td>
      <td>3546.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>sigma_alpha</td>
      <td>0.739</td>
      <td>0.043</td>
      <td>0.657</td>
      <td>0.821</td>
      <td>0.001</td>
      <td>0.000</td>
      <td>4525.0</td>
      <td>4519.0</td>
      <td>4516.0</td>
      <td>3116.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>sigma_beta</td>
      <td>1.273</td>
      <td>0.073</td>
      <td>1.142</td>
      <td>1.413</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>4252.0</td>
      <td>4240.0</td>
      <td>4245.0</td>
      <td>2588.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>sigma</td>
      <td>1.203</td>
      <td>0.010</td>
      <td>1.185</td>
      <td>1.221</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>7610.0</td>
      <td>7610.0</td>
      <td>7577.0</td>
      <td>2978.0</td>
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


![png](02_Hierarchical%20Model_files/02_Hierarchical%20Model_15_0.png)


## Non-Centered Version
See [Blog post](https://twiecki.io/blog/2017/02/08/bayesian-hierchical-non-centered/) by Thomas Wiecki


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
    Sampling 4 chains: 100%|██████████| 8000/8000 [02:09<00:00, 61.84draws/s] 
    The number of effective samples is smaller than 10% for some parameters.
    100%|██████████| 4000/4000 [00:07<00:00, 567.93it/s]



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
      <td>0.728</td>
      <td>0.255</td>
      <td>0.253</td>
      <td>1.210</td>
      <td>0.005</td>
      <td>0.003</td>
      <td>3002.0</td>
      <td>3002.0</td>
      <td>3034.0</td>
      <td>3393.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <td>beta_offset[1]</td>
      <td>0.859</td>
      <td>0.458</td>
      <td>0.021</td>
      <td>1.729</td>
      <td>0.006</td>
      <td>0.005</td>
      <td>5659.0</td>
      <td>4721.0</td>
      <td>5683.0</td>
      <td>2775.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <td>beta_offset[2]</td>
      <td>1.299</td>
      <td>0.314</td>
      <td>0.685</td>
      <td>1.848</td>
      <td>0.004</td>
      <td>0.003</td>
      <td>5039.0</td>
      <td>4444.0</td>
      <td>5059.0</td>
      <td>2614.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <td>beta_offset[3]</td>
      <td>-0.191</td>
      <td>0.974</td>
      <td>-2.100</td>
      <td>1.603</td>
      <td>0.009</td>
      <td>0.018</td>
      <td>11560.0</td>
      <td>1464.0</td>
      <td>11468.0</td>
      <td>2664.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <td>beta_offset[4]</td>
      <td>-0.078</td>
      <td>0.982</td>
      <td>-1.833</td>
      <td>1.795</td>
      <td>0.010</td>
      <td>0.017</td>
      <td>9256.0</td>
      <td>1604.0</td>
      <td>9275.0</td>
      <td>2977.0</td>
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
      <td>-0.712</td>
      <td>0.943</td>
      <td>-2.491</td>
      <td>1.063</td>
      <td>0.011</td>
      <td>0.011</td>
      <td>7782.0</td>
      <td>3507.0</td>
      <td>7774.0</td>
      <td>3113.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <td>mu_alpha</td>
      <td>3.641</td>
      <td>0.058</td>
      <td>3.535</td>
      <td>3.753</td>
      <td>0.002</td>
      <td>0.002</td>
      <td>614.0</td>
      <td>614.0</td>
      <td>612.0</td>
      <td>1224.0</td>
      <td>1.01</td>
    </tr>
    <tr>
      <td>sigma_beta</td>
      <td>1.275</td>
      <td>0.071</td>
      <td>1.150</td>
      <td>1.413</td>
      <td>0.003</td>
      <td>0.002</td>
      <td>591.0</td>
      <td>591.0</td>
      <td>582.0</td>
      <td>1351.0</td>
      <td>1.01</td>
    </tr>
    <tr>
      <td>sigma_alpha</td>
      <td>0.742</td>
      <td>0.043</td>
      <td>0.660</td>
      <td>0.821</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>1079.0</td>
      <td>1079.0</td>
      <td>1081.0</td>
      <td>1793.0</td>
      <td>1.01</td>
    </tr>
    <tr>
      <td>sigma</td>
      <td>1.203</td>
      <td>0.009</td>
      <td>1.185</td>
      <td>1.220</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>7527.0</td>
      <td>7527.0</td>
      <td>7521.0</td>
      <td>3081.0</td>
      <td>1.00</td>
    </tr>
  </tbody>
</table>
<p>443 rows × 11 columns</p>
</div>




```python
az.plot_trace(trace, var_names=["mu_alpha", "mu_beta", "sigma_alpha", "sigma_beta", "sigma"])
plt.show()
```


![png](02_Hierarchical%20Model_files/02_Hierarchical%20Model_19_0.png)


## Multivariate Model (includes correlation)


```python
with pm.Model() as correlation_hier_model:
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



    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    <ipython-input-10-6836c0bd6b81> in <module>
         27     #prior = pm.sample_prior_predictive()
         28     trace = pm.sample(random_seed=2412, chains=4, 
    ---> 29                       draws=1000, tune=1000)
         30     posterior_predictive = pm.sample_posterior_predictive(trace)


    ~/.pyenv/versions/anaconda3-2019.03/lib/python3.7/site-packages/pymc3/sampling.py in sample(draws, step, init, n_init, start, trace, chain_idx, chains, cores, tune, progressbar, model, random_seed, discard_tuned_samples, compute_convergence_checks, **kwargs)
        394                 start_, step = init_nuts(init=init, chains=chains, n_init=n_init,
        395                                          model=model, random_seed=random_seed,
    --> 396                                          progressbar=progressbar, **kwargs)
        397                 if start is None:
        398                     start = start_


    ~/.pyenv/versions/anaconda3-2019.03/lib/python3.7/site-packages/pymc3/sampling.py in init_nuts(init, chains, n_init, model, random_seed, progressbar, **kwargs)
       1513             'Unknown initializer: {}.'.format(init))
       1514 
    -> 1515     step = pm.NUTS(potential=potential, model=model, **kwargs)
       1516 
       1517     return start, step


    ~/.pyenv/versions/anaconda3-2019.03/lib/python3.7/site-packages/pymc3/step_methods/hmc/nuts.py in __init__(self, vars, max_treedepth, early_max_treedepth, **kwargs)
        150         `pm.sample` to the desired number of tuning steps.
        151         """
    --> 152         super().__init__(vars, **kwargs)
        153 
        154         self.max_treedepth = max_treedepth


    ~/.pyenv/versions/anaconda3-2019.03/lib/python3.7/site-packages/pymc3/step_methods/hmc/base_hmc.py in __init__(self, vars, scaling, step_scale, is_cov, model, blocked, potential, dtype, Emax, target_accept, gamma, k, t0, adapt_step_size, step_rand, **theano_kwargs)
         70         vars = inputvars(vars)
         71 
    ---> 72         super().__init__(vars, blocked=blocked, model=model, dtype=dtype, **theano_kwargs)
         73 
         74         self.adapt_step_size = adapt_step_size


    ~/.pyenv/versions/anaconda3-2019.03/lib/python3.7/site-packages/pymc3/step_methods/arraystep.py in __init__(self, vars, model, blocked, dtype, **theano_kwargs)
        226 
        227         func = model.logp_dlogp_function(
    --> 228             vars, dtype=dtype, **theano_kwargs)
        229 
        230         # handle edge case discovered in #2948


    ~/.pyenv/versions/anaconda3-2019.03/lib/python3.7/site-packages/pymc3/model.py in logp_dlogp_function(self, grad_vars, **kwargs)
        721         varnames = [var.name for var in grad_vars]
        722         extra_vars = [var for var in self.free_RVs if var.name not in varnames]
    --> 723         return ValueGradFunction(self.logpt, grad_vars, extra_vars, **kwargs)
        724 
        725     @property


    ~/.pyenv/versions/anaconda3-2019.03/lib/python3.7/site-packages/pymc3/model.py in __init__(self, cost, grad_vars, extra_vars, dtype, casting, **kwargs)
        460 
        461         self._theano_function = theano.function(
    --> 462             inputs, [self._cost_joined, grad], givens=givens, **kwargs)
        463 
        464     def set_extra_values(self, extra_vars):


    ~/.pyenv/versions/anaconda3-2019.03/lib/python3.7/site-packages/theano/compile/function.py in function(inputs, outputs, mode, updates, givens, no_default_updates, accept_inplace, name, rebuild_strict, allow_input_downcast, profile, on_unused_input)
        315                    on_unused_input=on_unused_input,
        316                    profile=profile,
    --> 317                    output_keys=output_keys)
        318     return fn


    ~/.pyenv/versions/anaconda3-2019.03/lib/python3.7/site-packages/theano/compile/pfunc.py in pfunc(params, outputs, mode, updates, givens, no_default_updates, accept_inplace, name, rebuild_strict, allow_input_downcast, profile, on_unused_input, output_keys)
        484                          accept_inplace=accept_inplace, name=name,
        485                          profile=profile, on_unused_input=on_unused_input,
    --> 486                          output_keys=output_keys)
        487 
        488 


    ~/.pyenv/versions/anaconda3-2019.03/lib/python3.7/site-packages/theano/compile/function_module.py in orig_function(inputs, outputs, mode, accept_inplace, name, profile, on_unused_input, output_keys)
       1837                   on_unused_input=on_unused_input,
       1838                   output_keys=output_keys,
    -> 1839                   name=name)
       1840         with theano.change_flags(compute_test_value="off"):
       1841             fn = m.create(defaults)


    ~/.pyenv/versions/anaconda3-2019.03/lib/python3.7/site-packages/theano/compile/function_module.py in __init__(self, inputs, outputs, mode, accept_inplace, function_builder, profile, on_unused_input, fgraph, output_keys, name)
       1517                         optimizer, inputs, outputs)
       1518                 else:
    -> 1519                     optimizer_profile = optimizer(fgraph)
       1520 
       1521                 end_optimizer = time.time()


    ~/.pyenv/versions/anaconda3-2019.03/lib/python3.7/site-packages/theano/gof/opt.py in __call__(self, fgraph)
        106 
        107         """
    --> 108         return self.optimize(fgraph)
        109 
        110     def add_requirements(self, fgraph):


    ~/.pyenv/versions/anaconda3-2019.03/lib/python3.7/site-packages/theano/gof/opt.py in optimize(self, fgraph, *args, **kwargs)
         95             orig = theano.tensor.basic.constant.enable
         96             theano.tensor.basic.constant.enable = False
    ---> 97             ret = self.apply(fgraph, *args, **kwargs)
         98         finally:
         99             theano.tensor.basic.constant.enable = orig


    ~/.pyenv/versions/anaconda3-2019.03/lib/python3.7/site-packages/theano/gof/opt.py in apply(self, fgraph)
        249                     nb_nodes_before = len(fgraph.apply_nodes)
        250                     t0 = time.time()
    --> 251                     sub_prof = optimizer.optimize(fgraph)
        252                     l.append(float(time.time() - t0))
        253                     sub_profs.append(sub_prof)


    ~/.pyenv/versions/anaconda3-2019.03/lib/python3.7/site-packages/theano/gof/opt.py in optimize(self, fgraph, *args, **kwargs)
         95             orig = theano.tensor.basic.constant.enable
         96             theano.tensor.basic.constant.enable = False
    ---> 97             ret = self.apply(fgraph, *args, **kwargs)
         98         finally:
         99             theano.tensor.basic.constant.enable = orig


    ~/.pyenv/versions/anaconda3-2019.03/lib/python3.7/site-packages/theano/gof/opt.py in apply(self, fgraph, start_from)
       2538                 nb = change_tracker.nb_imported
       2539                 t_opt = time.time()
    -> 2540                 sub_prof = gopt.apply(fgraph)
       2541                 time_opts[gopt] += time.time() - t_opt
       2542                 sub_profs.append(sub_prof)


    ~/.pyenv/versions/anaconda3-2019.03/lib/python3.7/site-packages/theano/gof/opt.py in apply(self, fgraph, start_from)
       2141                     continue
       2142                 current_node = node
    -> 2143                 nb += self.process_node(fgraph, node)
       2144             loop_t = time.time() - t0
       2145         finally:


    ~/.pyenv/versions/anaconda3-2019.03/lib/python3.7/site-packages/theano/gof/opt.py in process_node(self, fgraph, node, lopt)
       2032         lopt = lopt or self.local_opt
       2033         try:
    -> 2034             replacements = lopt.transform(node)
       2035         except Exception as e:
       2036             if self.failure_callback is not None:


    ~/.pyenv/versions/anaconda3-2019.03/lib/python3.7/site-packages/theano/tensor/opt.py in constant_folding(node)
       6514         impl = 'py'
       6515     thunk = node.op.make_thunk(node, storage_map, compute_map,
    -> 6516                                no_recycling=[], impl=impl)
       6517 
       6518     required = thunk()


    ~/.pyenv/versions/anaconda3-2019.03/lib/python3.7/site-packages/theano/gof/op.py in make_thunk(self, node, storage_map, compute_map, no_recycling, impl)
        953             try:
        954                 return self.make_c_thunk(node, storage_map, compute_map,
    --> 955                                          no_recycling)
        956             except (NotImplementedError, utils.MethodNotDefined):
        957                 # We requested the c code, so don't catch the error.


    ~/.pyenv/versions/anaconda3-2019.03/lib/python3.7/site-packages/theano/gof/op.py in make_c_thunk(self, node, storage_map, compute_map, no_recycling)
        856         _logger.debug('Trying CLinker.make_thunk')
        857         outputs = cl.make_thunk(input_storage=node_input_storage,
    --> 858                                 output_storage=node_output_storage)
        859         thunk, node_input_filters, node_output_filters = outputs
        860 


    ~/.pyenv/versions/anaconda3-2019.03/lib/python3.7/site-packages/theano/gof/cc.py in make_thunk(self, input_storage, output_storage, storage_map, keep_lock)
       1215         cthunk, module, in_storage, out_storage, error_storage = self.__compile__(
       1216             input_storage, output_storage, storage_map,
    -> 1217             keep_lock=keep_lock)
       1218 
       1219         res = _CThunk(cthunk, init_tasks, tasks, error_storage, module)


    ~/.pyenv/versions/anaconda3-2019.03/lib/python3.7/site-packages/theano/gof/cc.py in __compile__(self, input_storage, output_storage, storage_map, keep_lock)
       1155                                             output_storage,
       1156                                             storage_map,
    -> 1157                                             keep_lock=keep_lock)
       1158         return (thunk,
       1159                 module,


    ~/.pyenv/versions/anaconda3-2019.03/lib/python3.7/site-packages/theano/gof/cc.py in cthunk_factory(self, error_storage, in_storage, out_storage, storage_map, keep_lock)
       1622                 node.op.prepare_node(node, storage_map, None, 'c')
       1623             module = get_module_cache().module_from_key(
    -> 1624                 key=key, lnk=self, keep_lock=keep_lock)
       1625 
       1626         vars = self.inputs + self.outputs + self.orphans


    ~/.pyenv/versions/anaconda3-2019.03/lib/python3.7/site-packages/theano/gof/cmodule.py in module_from_key(self, key, lnk, keep_lock)
       1187             try:
       1188                 location = dlimport_workdir(self.dirname)
    -> 1189                 module = lnk.compile_cmodule(location)
       1190                 name = module.__file__
       1191                 assert name.startswith(location)


    ~/.pyenv/versions/anaconda3-2019.03/lib/python3.7/site-packages/theano/gof/cc.py in compile_cmodule(self, location)
       1525                 lib_dirs=self.lib_dirs(),
       1526                 libs=libs,
    -> 1527                 preargs=preargs)
       1528         except Exception as e:
       1529             e.args += (str(self.fgraph),)


    ~/.pyenv/versions/anaconda3-2019.03/lib/python3.7/site-packages/theano/gof/cmodule.py in compile_str(module_name, src_code, location, include_dirs, lib_dirs, libs, preargs, py_module, hide_symbols)
       2349 
       2350         try:
    -> 2351             p_out = output_subprocess_Popen(cmd)
       2352             compile_stderr = decode(p_out[1])
       2353         except Exception:


    ~/.pyenv/versions/anaconda3-2019.03/lib/python3.7/site-packages/theano/misc/windows.py in output_subprocess_Popen(command, **params)
         78     # we need to use communicate to make sure we don't deadlock around
         79     # the stdout/stderr pipe.
    ---> 80     out = p.communicate()
         81     return out + (p.returncode,)


    ~/.pyenv/versions/anaconda3-2019.03/lib/python3.7/subprocess.py in communicate(self, input, timeout)
        918 
        919             try:
    --> 920                 stdout, stderr = self._communicate(input, endtime, timeout)
        921             except KeyboardInterrupt:
        922                 # https://bugs.python.org/issue25942


    ~/.pyenv/versions/anaconda3-2019.03/lib/python3.7/subprocess.py in _communicate(self, input, endtime, orig_timeout)
       1656                         raise TimeoutExpired(self.args, orig_timeout)
       1657 
    -> 1658                     ready = selector.select(timeout)
       1659                     self._check_timeout(endtime, orig_timeout)
       1660 


    ~/.pyenv/versions/anaconda3-2019.03/lib/python3.7/selectors.py in select(self, timeout)
        413         ready = []
        414         try:
    --> 415             fd_event_list = self._selector.poll(timeout)
        416         except InterruptedError:
        417             return ready


    KeyboardInterrupt: 



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
      <td>3.644</td>
      <td>0.059</td>
      <td>3.536</td>
      <td>3.760</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>2606.0</td>
      <td>2606.0</td>
      <td>2611.0</td>
      <td>2554.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>mu[1]</td>
      <td>2.756</td>
      <td>0.100</td>
      <td>2.556</td>
      <td>2.938</td>
      <td>0.002</td>
      <td>0.001</td>
      <td>2777.0</td>
      <td>2777.0</td>
      <td>2782.0</td>
      <td>2769.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>ab[0,0]</td>
      <td>4.998</td>
      <td>0.163</td>
      <td>4.692</td>
      <td>5.310</td>
      <td>0.002</td>
      <td>0.002</td>
      <td>4450.0</td>
      <td>4444.0</td>
      <td>4456.0</td>
      <td>2811.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>ab[0,1]</td>
      <td>3.822</td>
      <td>0.306</td>
      <td>3.261</td>
      <td>4.403</td>
      <td>0.004</td>
      <td>0.003</td>
      <td>4858.0</td>
      <td>4783.0</td>
      <td>4832.0</td>
      <td>3083.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>ab[1,0]</td>
      <td>5.211</td>
      <td>0.296</td>
      <td>4.654</td>
      <td>5.768</td>
      <td>0.005</td>
      <td>0.003</td>
      <td>4010.0</td>
      <td>4010.0</td>
      <td>4018.0</td>
      <td>3041.0</td>
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
      <td>0.606</td>
      <td>0.615</td>
      <td>-0.476</td>
      <td>1.815</td>
      <td>0.008</td>
      <td>0.007</td>
      <td>5693.0</td>
      <td>3452.0</td>
      <td>5689.0</td>
      <td>3226.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>sigma[0]</td>
      <td>0.767</td>
      <td>0.045</td>
      <td>0.680</td>
      <td>0.850</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>2069.0</td>
      <td>2069.0</td>
      <td>2071.0</td>
      <td>2666.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>sigma[1]</td>
      <td>1.291</td>
      <td>0.074</td>
      <td>1.152</td>
      <td>1.431</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>2581.0</td>
      <td>2581.0</td>
      <td>2562.0</td>
      <td>2809.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>C_triu[0]</td>
      <td>0.715</td>
      <td>0.045</td>
      <td>0.628</td>
      <td>0.795</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>1571.0</td>
      <td>1568.0</td>
      <td>1576.0</td>
      <td>2760.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>sd_price</td>
      <td>1.203</td>
      <td>0.010</td>
      <td>1.184</td>
      <td>1.221</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>5778.0</td>
      <td>5778.0</td>
      <td>5787.0</td>
      <td>2824.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>444 rows × 11 columns</p>
</div>




```python

```
