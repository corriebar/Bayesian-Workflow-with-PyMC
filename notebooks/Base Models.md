

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pymc3 as pm

import altair as alt
import arviz as az
alt.data_transformers.disable_max_rows()
```

    /home/corrie/.pyenv/versions/anaconda3-2019.03/lib/python3.7/site-packages/xarray/core/merge.py:17: FutureWarning: The Panel class is removed from pandas. Accessing it from the top-level namespace will also be removed in the next version
      PANDAS_TYPES = (pd.Series, pd.DataFrame, pd.Panel)





    DataTransformerRegistry.enable('default')




```python
d = pd.read_csv("../data/interim_data/houses.csv", dtype = {"ags": str, "plz": str}, index_col=0)
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




    (8014, 20)




```python
alt.Chart(d).mark_bar().encode(
    alt.X("sqm_price:Q", 
            bin=alt.BinParams(maxbins=100)),
    y="count()"
)
```




![png](Base%20Models_files/Base%20Models_4_0.png)




```python
alt.Chart(d[d.price <= 1600000]).mark_bar().encode(
    alt.X("price:Q",
         bin = alt.BinParams(maxbins=100),
         scale = alt.Scale(domain=(0, 1600000)) ),
    y="count()"
)
```




![png](Base%20Models_files/Base%20Models_5_0.png)




```python
alt.Chart(d).mark_bar().encode(
    alt.X("log_price:Q",
         bin=alt.BinParams(maxbins=100)),
    y="count()"
)
```




![png](Base%20Models_files/Base%20Models_6_0.png)




```python
alt.Chart(d).mark_bar().encode(
    alt.X("log_sqm_price_s:Q", 
            bin=alt.BinParams(maxbins=100)),
    y="count()"
)
```




![png](Base%20Models_files/Base%20Models_7_0.png)



$$\begin{align*}
log(sqm/price) &\sim \text{Normal}(\mu, \sigma)\\
\mu &\sim \text{Normal}(0, 10)\\
\sigma &\sim \text{HalfCauchy}(10)
\end{align*}$$


```python
target = "log_price_s"
```


```python
with pm.Model() as intercept_normal:
    mu = pm.Normal("mu", mu=0, sd=10)
    sigma = pm.HalfNormal("sigma", sd=10)
    
    y = pm.Normal("y", mu=mu, sd=sigma, observed = d[target])
    
    trace = pm.sample()
```

    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (2 chains in 2 jobs)
    NUTS: [sigma, mu]
    Sampling 2 chains: 100%|██████████| 2000/2000 [00:01<00:00, 1017.84draws/s]
    The acceptance probability does not match the target. It is 0.8795999542168703, but should be close to 0.8. Try to increase the number of tuning steps.



```python
az.plot_trace(trace)
plt.show()
```


![png](Base%20Models_files/Base%20Models_11_0.png)



```python
with pm.Model() as intercept_student:
    mu = pm.Normal("mu", mu=0, sd=20)
    sigma = pm.HalfCauchy("sigma", 10)
    nu = pm.Gamma("nu", alpha=2, beta=0.1)
    
    y = pm.StudentT("y", mu=mu, sd=sigma, nu=nu, observed=d[target])
    
    trace = pm.sample()
```

    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (2 chains in 2 jobs)
    NUTS: [nu, sigma, mu]
    Sampling 2 chains: 100%|██████████| 2000/2000 [00:07<00:00, 283.54draws/s]



```python
az.plot_trace(trace)
plt.show()
```


![png](Base%20Models_files/Base%20Models_13_0.png)


$$\begin{align*}
log(sqm/price) &\sim \text{Normal}(\mu, \sigma)\\
\mu &= \alpha + \beta \;\text{living_space}\\
\alpha &\sim \text{Normal}(0, 10) \\
\beta &\sim \text{Normal}(0, 1) \\
\sigma &\sim \text{HalfCauchy}(10)
\end{align*}$$


```python
alt.Chart(d).mark_circle(clip=True,
                        opacity=0.7,
                        size=5).encode(
    x = alt.X("living_space", scale=alt.Scale(domain=(0, 400))),
    y = alt.Y("price", scale=alt.Scale(domain=(0, 1500000)))
)
```




![png](Base%20Models_files/Base%20Models_15_0.png)




```python
with pm.Model() as lin_normal:
    alpha = pm.Normal("alpha", mu=0, sd=10)
    beta = pm.Normal("beta", mu=0, sd=1)
    mu = alpha + beta*d["living_space_s"]
    
    sigma = pm.HalfCauchy("sigma", 5)
    y = pm.Normal("y", mu=mu, sd=sigma, observed=d[target])
    trace = pm.sample()
```

    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (2 chains in 2 jobs)
    NUTS: [sigma, beta, alpha]
    Sampling 2 chains: 100%|██████████| 2000/2000 [00:02<00:00, 682.73draws/s]



```python
az.plot_trace(trace)
plt.show()
```


![png](Base%20Models_files/Base%20Models_17_0.png)



```python
with pm.Model() as lin_student:
    alpha = pm.Normal("alpha", mu=0, sd=10)
    beta = pm.Normal("beta", mu=0, sd=1)
    mu = alpha + beta*d["living_space_s"]
    
    sigma = pm.HalfCauchy("sigma", 5)
    nu = pm.Gamma("nu", alpha=2, beta=0.1)
    y = pm.StudentT("y", nu=nu, mu=mu, sd=sigma, observed=d[target])
    trace = pm.sample(random_seed=2405)
```

    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (2 chains in 2 jobs)
    NUTS: [nu, sigma, beta, alpha]
    Sampling 2 chains: 100%|██████████| 2000/2000 [00:07<00:00, 283.71draws/s]



```python
az.plot_trace(trace)
plt.show()
```


![png](Base%20Models_files/Base%20Models_19_0.png)



```python

```
