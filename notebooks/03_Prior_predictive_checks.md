# Prior Predictive Checks
In this notebook, we compare different priors and what they mean for our model. In particular, we look at flat priors, weakly informative priors and informative priors.
If you want to know more about which priors and what prior parameters are recommended for which kind of model, check the [prior recommendation](https://github.com/stan-dev/stan/wiki/Prior-Choice-Recommendations) by the Stan team.


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
from utils.plot_utils import draw_model_plot, set_plot_defaults
```

    WARNING (theano.tensor.blas): Using NumPy C-API based implementation for BLAS functions.
    /home/corrie/.pyenv/versions/anaconda3-2019.03/lib/python3.7/site-packages/xarray/core/merge.py:17: FutureWarning: The Panel class is removed from pandas. Accessing it from the top-level namespace will also be removed in the next version
      PANDAS_TYPES = (pd.Series, pd.DataFrame, pd.Panel)



```python
d, _, _ = load_data()
set_plot_defaults("Europace Sans")
target = "price_s"
```

## Flat priors
The idea behind flat priors is to have uninformative priors and thus be (supposedly) more objective/less subjective.

Note that some people call only unproper priors as flat priors. These are priors that have equal probability density everywhere and are thus not a real probability distribution (and are hence called unproper priors). These unproper priors mostly still lead to a valid and proper posterior. The flat priors I'm using here are then sometimes called very vague prior.


```python
with pm.Model() as flat_prior:
    alpha = pm.Normal("alpha", mu=0, sd=100)
    beta = pm.Normal("beta", mu=0, sd=100)
    mu = alpha + beta*d["living_space_s"]
    
    sigma = pm.Exponential("sigma", 1/100)
    y = pm.Normal("y", mu=mu, sd=sigma, observed=d[target])
    
    flat_prior = pm.sample_prior_predictive()
    flat_trace = pm.sample(draws=50, tune=50) # this is only until this issue is fixed: https://github.com/arviz-devs/arviz/issues/822
                                              # since we're not fitting data for a prior predictive check, it's not necessary to sample
```

    Only 50 samples in chain.
    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (2 chains in 2 jobs)
    NUTS: [sigma, beta, alpha]
    Sampling 2 chains: 100%|██████████| 200/200 [00:00<00:00, 307.85draws/s]
    /home/corrie/.pyenv/versions/anaconda3-2019.03/lib/python3.7/site-packages/pymc3/sampling.py:464: UserWarning: The number of samples is too small to check convergence reliably.
      warnings.warn("The number of samples is too small to check convergence reliably.")
    The acceptance probability does not match the target. It is 0.913688804620866, but should be close to 0.8. Try to increase the number of tuning steps.


Just for comparison, a standard normal prior.


```python
with pm.Model() as normal_prior:             # I guess I also could have just used numpy to generate the distribution
    alpha = pm.Normal("alpha", mu=0, sd=1)
    beta = pm.Normal("beta", mu=0, sd=1)
    mu = alpha + beta*d["living_space_s"]
    
    sigma = pm.Exponential("sigma", 1)
    y = pm.Normal("y", mu=mu, sd=sigma, observed=d[target])
    
    normal_prior = pm.sample_prior_predictive()
    normal_trace = pm.sample(draws=50, tune=50)
```

    Only 50 samples in chain.
    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (2 chains in 2 jobs)
    NUTS: [sigma, beta, alpha]
    Sampling 2 chains: 100%|██████████| 200/200 [00:00<00:00, 457.87draws/s]
    /home/corrie/.pyenv/versions/anaconda3-2019.03/lib/python3.7/site-packages/pymc3/sampling.py:464: UserWarning: The number of samples is too small to check convergence reliably.
      warnings.warn("The number of samples is too small to check convergence reliably.")
    The acceptance probability does not match the target. It is 0.9745948272037334, but should be close to 0.8. Try to increase the number of tuning steps.
    The acceptance probability does not match the target. It is 0.9325072895897503, but should be close to 0.8. Try to increase the number of tuning steps.



```python
flat_data = az.from_pymc3(trace = flat_trace, prior = flat_prior )
normal_data = az.from_pymc3(trace = normal_trace, prior=normal_prior )
```

Visualizing the prior helps to understand what they mean.


```python
ax = az.plot_density(flat_data, group="prior",
                data_labels = ["flat"] ,
               var_names = ["alpha", "beta", "sigma"],
               shade=0.3, bw=8, figsize=(20,6), credible_interval=1,
                    colors="#f07855",  point_estimate=None)
ax[0].set_title("alpha", fontdict={'fontsize': 20})
ax[1].set_title("beta", fontdict={'fontsize': 20})
ax[2].set_title("sigma", fontdict={'fontsize': 20})
ax[0].tick_params("x", labelsize=16)
ax[1].tick_params("x", labelsize=16)
ax[2].tick_params("x", labelsize=16)
ax[0].set_xlabel(" ", fontdict={"fontsize": 18}, labelpad=15)
ax[1].set_xlabel(" ", fontdict={"fontsize": 18}, labelpad=15)
ax[2].set_xlabel(" ", fontdict={"fontsize": 18}, labelpad=15)
plt.suptitle("Flat Prior", fontsize=30)
plt.show()
```


![png](03_Prior_predictive_checks_files/03_Prior_predictive_checks_9_0.png)


When we compare this prior with a standard normal distribution, it is easy to see why this kind of prior is called flat.


```python
ax = az.plot_density([flat_data, normal_data], group="prior",
                data_labels = ["flat", "standard normal"] ,
               var_names = ["alpha", "beta", "sigma"],
               shade=0.3, bw=8, figsize=(20,6), credible_interval=1,
                    colors=["#f07855", "#cbdd92"],  point_estimate=None)
ax[0].legend(frameon=False, prop={'size': 18}, markerscale=3., loc="upper left")
ax[0].set_title("alpha", fontdict={'fontsize': 20})
ax[1].set_title("beta", fontdict={'fontsize': 20})
ax[2].set_title("sigma", fontdict={'fontsize': 20})
ax[0].set_xlabel(" ", fontdict={"fontsize": 18}, labelpad=15)
ax[1].set_xlabel(" ", fontdict={"fontsize": 18}, labelpad=15)
ax[2].set_xlabel(" ", fontdict={"fontsize": 18}, labelpad=15)
plt.suptitle("Flat Prior")
plt.show()
```


![png](03_Prior_predictive_checks_files/03_Prior_predictive_checks_11_0.png)


When doing prior checks, it is always very important to know the scale of your data. 
For this use case, I scaled the outcome variable by dividing by 100,000. That is, $y$ the price variable is the home price in 100,000€. 
Furthermore, I standardized the living area so that our intercept, the $\alpha$ parameter represents the price of an average home (in 100,000€).
Since $\beta$ is standardized, the slope parameter $\beta$ represents the change in the price if the living area changes by a standard deviation (that is 76sqm).


```python
ax = az.plot_density([flat_data, normal_data], group="prior",
                data_labels = ["flat", "standard normal"] ,
               var_names = ["alpha", "beta", "sigma"],
               shade=0.3, bw=8, figsize=(20,6), credible_interval=1,
                    colors=["#f07855", "#cbdd92"],  point_estimate=None)
ax[0].legend(frameon=False, prop={'size': 18}, markerscale=3., loc="upper left")
ax[0].set_title("alpha", fontdict={'fontsize': 20})
ax[1].set_title("beta", fontdict={'fontsize': 20})
ax[2].set_title("sigma", fontdict={'fontsize': 20})
ax[0].set_xlabel("[Price of avg house in 100,000€]", fontdict={"fontsize": 18}, labelpad=15)
ax[1].set_xlabel("[Price change in 100,000€]", fontdict={"fontsize": 18}, labelpad=15)
ax[2].set_xlabel("[Price spread in 100,000€]", fontdict={"fontsize": 18}, labelpad=15)
plt.suptitle("Flat Prior")
plt.show()
```


![png](03_Prior_predictive_checks_files/03_Prior_predictive_checks_13_0.png)


Since we're working with a simple linear model with only one predictor, the best way to understand these priors is to plot the resulting model.


```python
np.random.seed(2405)
fig, ax = draw_model_plot(flat_prior, title="Flat Prior")
plt.show()
```


![png](03_Prior_predictive_checks_files/03_Prior_predictive_checks_15_0.png)


Ideally, we should pick priors before seeing our without consulting our data. A good way to get the priors, is to make use of our external knowledge. We can for example google  what was the most expensive flat sold in Berlin. This gives a good reference point.


```python
fig, ax2 = draw_model_plot(flat_prior, title="Flat Prior")
ax2.axhline(y=8.5e6, c="#3d5167", label="Most expensive flat sold in Berlin", lw=3)
ax2.axhline(y=0, c="#f07855", label="0 €", lw=3)
ax2.legend(frameon=False, markerscale=3., fontsize=20)
#ax2.set_xlim(ax.get_xlim())
plt.show()
```


![png](03_Prior_predictive_checks_files/03_Prior_predictive_checks_17_0.png)


Our uninformed prior puts a lot of probability mass on implausible values. Michael Betancourt argues in his [case studies](https://mc-stan.org/users/documentation/case-studies/weakly_informative_shapes.html) that this means our priors are actually quite informative that pull the posterior towards extreme values and can thus (negatively) bias our inference. 

Computationally, it also means that the algorithm can take longer to converge since it unecessarily tries to explore values that are simply not plausible.

## Weakly informative prior

Many people nowadays argue for weakly informative priors instead of flat priors (see e.g. [Andrew Gelman](https://statmodeling.stat.columbia.edu/2007/05/11/weakly_informat/)). The idea is to capture the general scale of the data to give the algorithm some orientation without using overly specific priors. 

In our use case for example, we'd like the prior to put most probability mass between the reference values we identified above. The priors picked here are taken from the [default weakly informative priors](http://mc-stan.org/rstanarm/articles/priors.html) from the R package `rstanarm`.


```python
# using different seeds just to get nicer plots
np.random.seed(12)
```


```python
with pm.Model() as weakly_inf_prior:
    alpha = pm.Normal("alpha", mu=0, sd=20)
    beta = pm.Normal("beta", mu=0, sd=5)
    mu = alpha + beta*d["living_space_s"]
    
    sigma = pm.Exponential("sigma", lam = 1/2.5)
    y = pm.Normal("y", mu=mu, sd=sigma, observed=d[target])
    
    weakly_inf_prior = pm.sample_prior_predictive()
    weak_trace = pm.sample(draws=50, tune=50)
```

    Only 50 samples in chain.
    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (2 chains in 2 jobs)
    NUTS: [sigma, beta, alpha]
    Sampling 2 chains: 100%|██████████| 200/200 [00:00<00:00, 538.28draws/s]
    /home/corrie/.pyenv/versions/anaconda3-2019.03/lib/python3.7/site-packages/pymc3/sampling.py:464: UserWarning: The number of samples is too small to check convergence reliably.
      warnings.warn("The number of samples is too small to check convergence reliably.")
    The acceptance probability does not match the target. It is 0.9771716690694046, but should be close to 0.8. Try to increase the number of tuning steps.



```python
weakly_inf_data = az.from_pymc3(trace = weak_trace, prior=weakly_inf_prior)
```


```python
ax = az.plot_density(weakly_inf_data, group="prior",
                data_labels = ["weakly informative prior"],
               var_names = ["alpha", "beta", "sigma"],
               shade=0.3, bw=8, figsize=(20,6), credible_interval=1,
                    colors="#3d5167", point_estimate=None)
ax[0].set_xlim(-60, 60)
ax[1].set_xlim(-15, 15)
ax[2].set_xlim(0, 16)
ax[0].set_title("alpha", fontdict={'fontsize': 20})
ax[1].set_title("beta", fontdict={'fontsize': 20})
ax[2].set_title("sigma", fontdict={'fontsize': 20})
ax[0].set_xlabel("[Price of avg house in 100,000€]", fontdict={"fontsize": 18}, labelpad=15)
ax[1].set_xlabel("[Price change in 100,000€]", fontdict={"fontsize": 18}, labelpad=15)
ax[2].set_xlabel("[Price spread in 100,000€]", fontdict={"fontsize": 18}, labelpad=15)
plt.suptitle("Weakly Informative Prior")
plt.show()
```


![png](03_Prior_predictive_checks_files/03_Prior_predictive_checks_23_0.png)


Comparing this with the flat prior from before, this prior covers a much smaller range of values.


```python
axes = az.plot_density([weakly_inf_data, flat_data], group="prior",
                data_labels = ["weakly\ninformative", "flat"],
               var_names = ["alpha", "beta", "sigma"],
               shade=0.3, bw=8, figsize=(20,6), credible_interval=1, point_estimate=None,
                      colors=["#3d5167", "#f07855"])
axes[0].set_xlim(-60, 60)
axes[1].set_xlim(-15, 15)
axes[2].set_xlim(0, 16)
axes[0].legend(frameon=False, prop={'size': 18}, markerscale=3., loc="upper left")
axes[0].set_title("alpha", fontdict={'fontsize': 20})
axes[1].set_title("beta", fontdict={'fontsize': 20})
axes[2].set_title("sigma", fontdict={'fontsize': 20})
axes[0].set_xlabel("[Price of avg house in 100,000€]", fontdict={"fontsize": 18}, labelpad=15)
axes[1].set_xlabel("[Price change in 100,000€]", fontdict={"fontsize": 18}, labelpad=15)
axes[2].set_xlabel("[Price spread in 100,000€]", fontdict={"fontsize": 18}, labelpad=15)
plt.suptitle("Weakly Informative Prior")
plt.show()
```


![png](03_Prior_predictive_checks_files/03_Prior_predictive_checks_25_0.png)



```python
np.random.seed(1) # to get nicer plots
fig, ax = draw_model_plot(weakly_inf_prior, title="Weakly Informative Prior")
ax.axhline(y=8.5e6, c="#3d5167", label="Most expensive flat sold in Berlin", lw=3)
ax.axhline(y=0, c="#f07855", label="0 €", lw=3)
ax.legend(frameon=False, markerscale=3., fontsize=20)
plt.show()
```


![png](03_Prior_predictive_checks_files/03_Prior_predictive_checks_26_0.png)


The models induced by this prior stays mostly within the scale of our reference points. It still puts a lot of probability mass on negative values (which we know can't happen) and also many models with a negative slope (meaning the homes get cheaper when they get bigger, unlikely to happen).
We could use this knowledge and generate an informed prior that takes this into account:


```python
with pm.Model() as inf_prior:
    alpha = pm.Normal("alpha", mu=3, sd=2.5)
    beta = pm.Normal("beta", mu=1, sd=2)
    mu = alpha + beta*d["living_space_s"]
    
    sigma = pm.Exponential("sigma", lam = 1/2)
    y = pm.Normal("y", mu=mu, sd=sigma, observed=d[target])
    
    inf_prior = pm.sample_prior_predictive()
    inf_trace = pm.sample(draws=50, tune=50)
```

    Only 50 samples in chain.
    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (2 chains in 2 jobs)
    NUTS: [sigma, beta, alpha]
    Sampling 2 chains: 100%|██████████| 200/200 [00:00<00:00, 499.41draws/s]
    /home/corrie/.pyenv/versions/anaconda3-2019.03/lib/python3.7/site-packages/pymc3/sampling.py:464: UserWarning: The number of samples is too small to check convergence reliably.
      warnings.warn("The number of samples is too small to check convergence reliably.")
    The acceptance probability does not match the target. It is 0.9413059770218641, but should be close to 0.8. Try to increase the number of tuning steps.
    The acceptance probability does not match the target. It is 0.9413402582684877, but should be close to 0.8. Try to increase the number of tuning steps.



```python
inf_data = az.from_pymc3(trace = inf_trace, prior=inf_prior)
```


```python
ax = az.plot_density(inf_data, group="prior",
                data_labels = ["informative prior"],
                  var_names = ["alpha", "beta", "sigma"],
               shade=0.4, bw=8, figsize=(20,6), credible_interval=1,
                    colors="#abd8ce", point_estimate=None)
ax[0].set_title("alpha", fontdict={'fontsize': 20})
ax[1].set_title("beta", fontdict={'fontsize': 20})
ax[2].set_title("sigma", fontdict={'fontsize': 20})
ax[0].set_xlabel("[Price of avg house in 100,000€]", fontdict={"fontsize": 18}, labelpad=15)
ax[1].set_xlabel("[Price change in 100,000€]", fontdict={"fontsize": 18}, labelpad=15)
ax[2].set_xlabel("[Price spread in 100,000€]", fontdict={"fontsize": 18}, labelpad=15)
plt.suptitle("Informative Prior")
plt.show()
```


![png](03_Prior_predictive_checks_files/03_Prior_predictive_checks_30_0.png)



```python
ax = az.plot_density([ weakly_inf_data, inf_data], group="prior",
                data_labels = ["weakly\ninformative", "informative"],
                  var_names = ["alpha", "beta", "sigma"],
               shade=0.4, bw=8, figsize=(20,6), credible_interval=1,
                    colors=["#3d5167","#abd8ce"], point_estimate=None)
ax[0].set_title("alpha", fontdict={'fontsize': 20})
ax[1].set_title("beta", fontdict={'fontsize': 20})
ax[2].set_title("sigma", fontdict={'fontsize': 20})
ax[0].set_xlabel("[Price of avg house in 100,000€]", fontdict={"fontsize": 18}, labelpad=15)
ax[1].set_xlabel("[Price change in 100,000€]", fontdict={"fontsize": 18}, labelpad=15)
ax[2].set_xlabel("[Price spread in 100,000€]", fontdict={"fontsize": 18}, labelpad=15)
ax[0].legend(frameon=False, prop={'size': 18}, markerscale=3., loc="upper left")
plt.suptitle("Informative Prior")
plt.show()
```


![png](03_Prior_predictive_checks_files/03_Prior_predictive_checks_31_0.png)



```python
fig, ax = draw_model_plot(inf_prior, title="Informative Prior")
#ax.axhline(y=8.5e6, c="#3d5167", label="Most expensive flat sold in Berlin", lw=3)
ax.axhline(y=0, c="#f07855", label="0 €", lw=3)
ax.legend(frameon=False)
plt.show()
```


![png](03_Prior_predictive_checks_files/03_Prior_predictive_checks_32_0.png)


Even in this informed prior, we still allow the model to vary by quite a bit. There are still a few lines with a negative slope, but overall most slopes are positive and also above the 0-line.
