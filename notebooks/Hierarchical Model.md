
# Hierarchical Model


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pymc3 as pm
import theano.tensor as tt

import altair as alt
import arviz as az
alt.data_transformers.disable_max_rows()

def destandardize_area(area_s):
    return area_s*np.std(d["living_space"]) + d["living_space"].mean()

def destandardize_price(price_s):
    return price_s * 100000
```

    /home/corrie/.pyenv/versions/anaconda3-2019.03/lib/python3.7/site-packages/xarray/core/merge.py:17: FutureWarning: The Panel class is removed from pandas. Accessing it from the top-level namespace will also be removed in the next version
      PANDAS_TYPES = (pd.Series, pd.DataFrame, pd.Panel)



```python
def draw_model_plot(sample, draws=50, title=""):
    draws = np.random.choice(len(sample["alpha"]), replace=False, size=draws)
    alpha = sample["alpha"][draws]
    beta = sample["beta"][draws]

    mu = alpha + beta*area_s[:, np.newaxis]
      
    fig, ax = plt.subplots(figsize=(13, 13))
    ax.plot(destandardize_area(area_s), destandardize_price(mu), c="gray", alpha=0.5)
    ax.set_xlabel("Living Area (in sqm)")
    ax.set_ylabel("Price (in €)")
    ax.set_title(title, fontdict={'fontsize': 30})
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
    return fig, ax
```


```python
d = pd.read_csv("../data/interim_data/houses.csv", dtype = {"ags": str, "zip": str}, index_col=0)
```


```python
target = "price_s"
```


```python
d.zip = d.zip.map(str.strip)
zip_codes = np.sort(d.zip.unique())
num_zip_codes = len(zip_codes)
zip_lookup = dict(zip(zip_codes, range(num_zip_codes)))
d["zip_code"] = d.zip.replace(zip_lookup).values
```

## Naive Model


```python
with pm.Model() as hier_model_naiv:
    mu_alpha = pm.Normal("mu_alpha", mu=3, sd=2.5)
    sigma_alpha = pm.Exponential("sigma_alpha", lam=1/2.5)
    
    mu_beta = pm.Normal("mu_beta", mu=1, sd=2.5)
    sigma_beta = pm.Exponential("sigma_beta", lam=1/2.5)
    
    alpha = pm.Normal("alpha", mu=mu_alpha, sd=sigma_alpha, shape=num_zip_codes)
    beta = pm.Normal("beta", mu=mu_beta, sd=sigma_beta, shape=num_zip_codes)
    # sigma also varies by zip code
    sigma = pm.Exponential("sigma", lam=1/2.5, shape=num_zip_codes)
    
    mu = alpha[d.zip_code.values] + beta[d.zip_code.values]*d.living_space_s
    y = pm.Normal("y", mu=mu, sd=sigma[d.zip_code.values], observed=d[target])
    
    prior = pm.sample_prior_predictive()
    trace = pm.sample(random_seed=2412, chains=4, 
                      draws=2000, tune=1000)
    posterior_predictive = pm.sample_posterior_predictive(trace)
```

    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (4 chains in 2 jobs)
    NUTS: [sigma, beta, alpha, sigma_beta, mu_beta, sigma_alpha, mu_alpha]
    Sampling 4 chains: 100%|██████████| 12000/12000 [07:43<00:00, 25.90draws/s] 
    There were 1062 divergences after tuning. Increase `target_accept` or reparameterize.
    The acceptance probability does not match the target. It is 0.5466976137891082, but should be close to 0.8. Try to increase the number of tuning steps.
    There were 806 divergences after tuning. Increase `target_accept` or reparameterize.
    The acceptance probability does not match the target. It is 0.6189983700147766, but should be close to 0.8. Try to increase the number of tuning steps.
    There were 823 divergences after tuning. Increase `target_accept` or reparameterize.
    The acceptance probability does not match the target. It is 0.6416072133904105, but should be close to 0.8. Try to increase the number of tuning steps.
    There were 802 divergences after tuning. Increase `target_accept` or reparameterize.
    The acceptance probability does not match the target. It is 0.6516621130960099, but should be close to 0.8. Try to increase the number of tuning steps.
    The estimated number of effective samples is smaller than 200 for some parameters.
    100%|██████████| 8000/8000 [00:11<00:00, 691.94it/s]



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
      <td>3.601</td>
      <td>0.055</td>
      <td>3.498</td>
      <td>3.704</td>
      <td>0.002</td>
      <td>0.001</td>
      <td>872.0</td>
      <td>871.0</td>
      <td>879.0</td>
      <td>2288.0</td>
      <td>1.01</td>
    </tr>
    <tr>
      <td>mu_beta</td>
      <td>2.694</td>
      <td>0.096</td>
      <td>2.483</td>
      <td>2.845</td>
      <td>0.013</td>
      <td>0.009</td>
      <td>57.0</td>
      <td>57.0</td>
      <td>67.0</td>
      <td>18.0</td>
      <td>1.04</td>
    </tr>
    <tr>
      <td>alpha[0]</td>
      <td>4.966</td>
      <td>0.158</td>
      <td>4.639</td>
      <td>5.245</td>
      <td>0.005</td>
      <td>0.004</td>
      <td>1006.0</td>
      <td>1006.0</td>
      <td>998.0</td>
      <td>744.0</td>
      <td>1.02</td>
    </tr>
    <tr>
      <td>alpha[1]</td>
      <td>5.152</td>
      <td>0.267</td>
      <td>4.608</td>
      <td>5.616</td>
      <td>0.015</td>
      <td>0.011</td>
      <td>332.0</td>
      <td>320.0</td>
      <td>319.0</td>
      <td>2805.0</td>
      <td>1.01</td>
    </tr>
    <tr>
      <td>alpha[2]</td>
      <td>5.054</td>
      <td>0.238</td>
      <td>4.584</td>
      <td>5.480</td>
      <td>0.010</td>
      <td>0.007</td>
      <td>549.0</td>
      <td>542.0</td>
      <td>582.0</td>
      <td>2821.0</td>
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
      <td>1.974</td>
      <td>1.896</td>
      <td>0.014</td>
      <td>5.277</td>
      <td>0.091</td>
      <td>0.064</td>
      <td>435.0</td>
      <td>435.0</td>
      <td>318.0</td>
      <td>90.0</td>
      <td>1.03</td>
    </tr>
    <tr>
      <td>sigma[215]</td>
      <td>0.837</td>
      <td>0.884</td>
      <td>0.014</td>
      <td>2.406</td>
      <td>0.034</td>
      <td>0.024</td>
      <td>687.0</td>
      <td>687.0</td>
      <td>345.0</td>
      <td>813.0</td>
      <td>1.02</td>
    </tr>
    <tr>
      <td>sigma[216]</td>
      <td>1.560</td>
      <td>1.444</td>
      <td>0.016</td>
      <td>4.171</td>
      <td>0.070</td>
      <td>0.049</td>
      <td>430.0</td>
      <td>430.0</td>
      <td>229.0</td>
      <td>401.0</td>
      <td>1.01</td>
    </tr>
    <tr>
      <td>sigma[217]</td>
      <td>1.992</td>
      <td>1.888</td>
      <td>0.013</td>
      <td>5.734</td>
      <td>0.188</td>
      <td>0.133</td>
      <td>101.0</td>
      <td>101.0</td>
      <td>163.0</td>
      <td>118.0</td>
      <td>1.03</td>
    </tr>
    <tr>
      <td>sigma[218]</td>
      <td>3.834</td>
      <td>2.295</td>
      <td>0.052</td>
      <td>7.531</td>
      <td>0.108</td>
      <td>0.077</td>
      <td>450.0</td>
      <td>450.0</td>
      <td>448.0</td>
      <td>431.0</td>
      <td>1.02</td>
    </tr>
  </tbody>
</table>
<p>661 rows × 11 columns</p>
</div>




```python
az.plot_trace(trace, var_names=["mu_alpha", "mu_beta", "sigma_alpha", "sigma_beta"])
plt.show()
```


![png](Hierarchical%20Model_files/Hierarchical%20Model_9_0.png)


## Centered Parameterization
In some cases, this actually performs better than the non-centered parameterization, see this [discussion](https://discourse.mc-stan.org/t/centered-vs-non-centered-parameterizations/7344)


```python
with pm.Model() as centered_hier_model:
    mu_alpha = pm.Normal("mu_alpha", mu=3, sd=2.5)
    sigma_alpha = pm.Exponential("sigma_alpha", lam=1/5)
    
    mu_beta = pm.Normal("mu_beta", mu=1, sd=2.5)
    sigma_beta = pm.Exponential("sigma_beta", lam=1/5)
    
    alpha = pm.Normal("alpha", mu=mu_alpha, sd=sigma_alpha, shape=num_zip_codes)
    beta = pm.Normal("beta", mu=mu_beta, sd=sigma_beta, shape=num_zip_codes)
    # without varying sigma
    sigma = pm.Exponential("sigma", lam=1/5)
    
    mu = alpha[d.zip_code.values] + beta[d.zip_code.values]*d.living_space_s
    y = pm.Normal("y", mu=mu, sd=sigma, observed=d[target])
    
    prior = pm.sample_prior_predictive()
    trace = pm.sample(random_seed=2412, chains=4, 
                      draws=2000, tune=1000)
    posterior_predictive = pm.sample_posterior_predictive(trace)
```

    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (4 chains in 2 jobs)
    NUTS: [sigma, beta, alpha, sigma_beta, mu_beta, sigma_alpha, mu_alpha]
    Sampling 4 chains: 100%|██████████| 12000/12000 [01:30<00:00, 133.12draws/s]
    100%|██████████| 8000/8000 [00:11<00:00, 709.31it/s]



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




    array([[<matplotlib.axes._subplots.AxesSubplot object at 0x7f86eca2fac8>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7f86ec9d9588>]],
          dtype=object)




![png](Hierarchical%20Model_files/Hierarchical%20Model_13_1.png)



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
      <td>3.640</td>
      <td>0.056</td>
      <td>3.534</td>
      <td>3.746</td>
      <td>0.001</td>
      <td>0.000</td>
      <td>11608.0</td>
      <td>11597.0</td>
      <td>11628.0</td>
      <td>6478.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>mu_beta</td>
      <td>2.773</td>
      <td>0.098</td>
      <td>2.583</td>
      <td>2.953</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>11421.0</td>
      <td>11381.0</td>
      <td>11446.0</td>
      <td>6370.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>alpha[0]</td>
      <td>4.977</td>
      <td>0.164</td>
      <td>4.676</td>
      <td>5.286</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>12831.0</td>
      <td>12817.0</td>
      <td>12853.0</td>
      <td>6410.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>alpha[1]</td>
      <td>5.099</td>
      <td>0.287</td>
      <td>4.547</td>
      <td>5.626</td>
      <td>0.003</td>
      <td>0.002</td>
      <td>12534.0</td>
      <td>12449.0</td>
      <td>12535.0</td>
      <td>6473.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>alpha[2]</td>
      <td>5.130</td>
      <td>0.200</td>
      <td>4.774</td>
      <td>5.517</td>
      <td>0.002</td>
      <td>0.001</td>
      <td>14125.0</td>
      <td>14125.0</td>
      <td>14127.0</td>
      <td>6122.0</td>
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
      <td>1.300</td>
      <td>0.103</td>
      <td>5.002</td>
      <td>0.010</td>
      <td>0.009</td>
      <td>16169.0</td>
      <td>11261.0</td>
      <td>16157.0</td>
      <td>5446.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>beta[218]</td>
      <td>0.503</td>
      <td>0.789</td>
      <td>-1.002</td>
      <td>1.959</td>
      <td>0.007</td>
      <td>0.008</td>
      <td>12164.0</td>
      <td>5440.0</td>
      <td>12204.0</td>
      <td>6845.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>sigma_alpha</td>
      <td>0.740</td>
      <td>0.043</td>
      <td>0.660</td>
      <td>0.821</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>8661.0</td>
      <td>8644.0</td>
      <td>8650.0</td>
      <td>6572.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>sigma_beta</td>
      <td>1.274</td>
      <td>0.073</td>
      <td>1.135</td>
      <td>1.408</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>8552.0</td>
      <td>8552.0</td>
      <td>8451.0</td>
      <td>5586.0</td>
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
      <td>16437.0</td>
      <td>16437.0</td>
      <td>16363.0</td>
      <td>5906.0</td>
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


![png](Hierarchical%20Model_files/Hierarchical%20Model_16_0.png)


## Non-Centered Version
See [Blog post](https://twiecki.io/blog/2017/02/08/bayesian-hierchical-non-centered/) by Thomas Wiecki


```python
with pm.Model() as non_centered_hier:
    
    beta_offset = pm.Normal("beta_offset", mu=0, sd=1, shape=num_zip_codes)
    
    mu_beta = pm.Normal("mu_beta", mu=0, sd=2.5)
    sigma_beta = pm.Exponential("sigma_beta", 1/5)
    beta = mu_beta + beta_offset*sigma_beta
    
    alpha_offset = pm.Normal("alpha_offset", mu=0, sd=1, shape=num_zip_codes)
    
    mu_alpha = pm.Normal("mu_alpha", mu=0, sd=2.5)
    sigma_alpha = pm.Exponential("sigma_alpha", 1/5)
    alpha = mu_alpha + alpha_offset*sigma_alpha
    
    mu = alpha[d.zip_code.values] + beta[d.zip_code.values]*d.living_space_s
    sigma = pm.Exponential("sigma", 1/5)
    y = pm.Normal("y", mu=mu, sd=sigma, observed=d[target])
    
    prior = pm.sample_prior_predictive()
    trace = pm.sample(random_seed=2412, chains=4, 
                      draws=2000, tune=1000)
    posterior_predictive = pm.sample_posterior_predictive(trace)
```

    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (4 chains in 2 jobs)
    NUTS: [sigma, sigma_alpha, mu_alpha, alpha_offset, sigma_beta, mu_beta, beta_offset]
    Sampling 4 chains: 100%|██████████| 12000/12000 [02:35<00:00, 77.15draws/s] 
    The number of effective samples is smaller than 10% for some parameters.
    100%|██████████| 8000/8000 [00:11<00:00, 673.09it/s]



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
      <td>0.735</td>
      <td>0.255</td>
      <td>0.266</td>
      <td>1.218</td>
      <td>0.003</td>
      <td>0.002</td>
      <td>6219.0</td>
      <td>6219.0</td>
      <td>6209.0</td>
      <td>6200.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>beta_offset[1]</td>
      <td>0.865</td>
      <td>0.460</td>
      <td>0.024</td>
      <td>1.751</td>
      <td>0.004</td>
      <td>0.003</td>
      <td>12188.0</td>
      <td>9534.0</td>
      <td>12197.0</td>
      <td>5688.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>beta_offset[2]</td>
      <td>1.302</td>
      <td>0.315</td>
      <td>0.729</td>
      <td>1.909</td>
      <td>0.003</td>
      <td>0.002</td>
      <td>9425.0</td>
      <td>8720.0</td>
      <td>9432.0</td>
      <td>5343.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>beta_offset[3]</td>
      <td>-0.199</td>
      <td>0.968</td>
      <td>-2.032</td>
      <td>1.610</td>
      <td>0.007</td>
      <td>0.012</td>
      <td>21284.0</td>
      <td>3093.0</td>
      <td>21196.0</td>
      <td>5302.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>beta_offset[4]</td>
      <td>-0.088</td>
      <td>0.981</td>
      <td>-1.820</td>
      <td>1.798</td>
      <td>0.007</td>
      <td>0.013</td>
      <td>20911.0</td>
      <td>3061.0</td>
      <td>20929.0</td>
      <td>5828.0</td>
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
      <td>alpha_offset[218]</td>
      <td>-0.721</td>
      <td>0.939</td>
      <td>-2.529</td>
      <td>1.010</td>
      <td>0.008</td>
      <td>0.008</td>
      <td>15183.0</td>
      <td>6859.0</td>
      <td>15175.0</td>
      <td>6505.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>mu_alpha</td>
      <td>3.641</td>
      <td>0.057</td>
      <td>3.536</td>
      <td>3.751</td>
      <td>0.002</td>
      <td>0.001</td>
      <td>1264.0</td>
      <td>1264.0</td>
      <td>1262.0</td>
      <td>2593.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>sigma_beta</td>
      <td>1.272</td>
      <td>0.071</td>
      <td>1.137</td>
      <td>1.403</td>
      <td>0.002</td>
      <td>0.001</td>
      <td>1290.0</td>
      <td>1290.0</td>
      <td>1273.0</td>
      <td>2696.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>sigma_alpha</td>
      <td>0.742</td>
      <td>0.043</td>
      <td>0.660</td>
      <td>0.822</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>2037.0</td>
      <td>2037.0</td>
      <td>2036.0</td>
      <td>3905.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>sigma</td>
      <td>1.203</td>
      <td>0.010</td>
      <td>1.185</td>
      <td>1.222</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>15606.0</td>
      <td>15604.0</td>
      <td>15647.0</td>
      <td>5705.0</td>
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


![png](Hierarchical%20Model_files/Hierarchical%20Model_20_0.png)


## Multivariate Model (includes correlation)


```python
with pm.Model() as correlation_hier_model:
    # mu is mean vector for alpha and beta parameter
    mu = pm.Normal("mu", mu=0, sd=2.5, shape=(2,)) 
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
                      draws=2000, tune=1000)
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
    Sampling 4 chains:   0%|          | 0/12000 [00:00<?, ?draws/s]/home/corrie/.pyenv/versions/anaconda3-2019.03/lib/python3.7/site-packages/theano/tensor/basic.py:6611: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      result[diagonal_slice] = x
    /home/corrie/.pyenv/versions/anaconda3-2019.03/lib/python3.7/site-packages/theano/tensor/basic.py:6611: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      result[diagonal_slice] = x
    Sampling 4 chains:  49%|████▉     | 5907/12000 [02:47<02:18, 44.06draws/s] /home/corrie/.pyenv/versions/anaconda3-2019.03/lib/python3.7/site-packages/theano/tensor/basic.py:6611: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      result[diagonal_slice] = x
    Sampling 4 chains:  50%|█████     | 6008/12000 [02:51<03:03, 32.67draws/s]/home/corrie/.pyenv/versions/anaconda3-2019.03/lib/python3.7/site-packages/theano/tensor/basic.py:6611: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      result[diagonal_slice] = x
    Sampling 4 chains: 100%|██████████| 12000/12000 [05:50<00:00, 34.24draws/s]
    100%|██████████| 8000/8000 [00:09<00:00, 806.44it/s]



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
      <td>3.645</td>
      <td>0.058</td>
      <td>3.535</td>
      <td>3.756</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>5436.0</td>
      <td>5436.0</td>
      <td>5402.0</td>
      <td>5317.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>mu[1]</td>
      <td>2.756</td>
      <td>0.099</td>
      <td>2.563</td>
      <td>2.937</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>5671.0</td>
      <td>5671.0</td>
      <td>5678.0</td>
      <td>5811.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>ab[0,0]</td>
      <td>4.994</td>
      <td>0.163</td>
      <td>4.688</td>
      <td>5.303</td>
      <td>0.002</td>
      <td>0.001</td>
      <td>9302.0</td>
      <td>9283.0</td>
      <td>9288.0</td>
      <td>6094.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>ab[0,1]</td>
      <td>3.815</td>
      <td>0.307</td>
      <td>3.233</td>
      <td>4.378</td>
      <td>0.003</td>
      <td>0.002</td>
      <td>10368.0</td>
      <td>10262.0</td>
      <td>10361.0</td>
      <td>6435.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>ab[1,0]</td>
      <td>5.212</td>
      <td>0.300</td>
      <td>4.668</td>
      <td>5.797</td>
      <td>0.003</td>
      <td>0.002</td>
      <td>8192.0</td>
      <td>8192.0</td>
      <td>8216.0</td>
      <td>6141.0</td>
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
      <td>0.610</td>
      <td>0.620</td>
      <td>-0.581</td>
      <td>1.751</td>
      <td>0.006</td>
      <td>0.005</td>
      <td>11890.0</td>
      <td>6796.0</td>
      <td>11896.0</td>
      <td>5857.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>sigma[0]</td>
      <td>0.767</td>
      <td>0.044</td>
      <td>0.682</td>
      <td>0.850</td>
      <td>0.001</td>
      <td>0.000</td>
      <td>4596.0</td>
      <td>4596.0</td>
      <td>4551.0</td>
      <td>6045.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>sigma[1]</td>
      <td>1.291</td>
      <td>0.073</td>
      <td>1.152</td>
      <td>1.427</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>5497.0</td>
      <td>5475.0</td>
      <td>5549.0</td>
      <td>6006.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>C_triu[0]</td>
      <td>0.715</td>
      <td>0.045</td>
      <td>0.632</td>
      <td>0.795</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>3509.0</td>
      <td>3502.0</td>
      <td>3510.0</td>
      <td>5514.0</td>
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
      <td>11588.0</td>
      <td>11588.0</td>
      <td>11546.0</td>
      <td>5547.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>444 rows × 11 columns</p>
</div>




```python

```
