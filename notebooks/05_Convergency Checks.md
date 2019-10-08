# Convergency checks
In this model, we use ArviZ to check if our model has converged.


```python
import sys
sys.path.append('../src/')

import pymc3 as pm
import arviz as az
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from utils.data_utils import load_data
from utils.plot_utils import set_plot_defaults
from utils.convergence_utils import check_mcse, check_neff, check_rhat
```


```python
set_plot_defaults(font="Europace Sans")
plt.rcParams['font.size'] = 16
plt.rcParams['axes.titlesize'] = 20
d, zip_lookup, num_zip_codes = load_data(kind="prices")   # loads data from data/interim_data/houses.csv 
                                                          # aternatively, use kind="rents" to load data from data/interim_data/rent.csv
zip_codes = np.sort(d.zip.unique())
target = "price_s"
```

We will check both the centered hierarchical model as well as the bad model, just to see how a model looks like that did not converge.


```python
data = az.from_netcdf("../models/centered_hier.nc")
data
```


```python
bad_data = az.from_netcdf("../models/bad_model.nc")
```


```python
az.summary(data.posterior)
```

## Trace plots
The first thing to check to see if a model converged or not are the trace plots. If you have more than 2 chains, it can be a good idea to only look at two, otherwise the plots can get very crowded.


```python
az.plot_trace(data, var_names=["mu_alpha", "mu_beta", "sigma_alpha", "sigma_beta"],
             coords={"chain":[0,1]})
plt.suptitle("Trace plots")
plt.show()
```

These trace plots look goot but we only look here at a selection of parameters. Remember that for each ZIP code, we fit both an $\alpha$ and $\beta$ parameter.
This means we actually have hundreds of parameter and we cannot feasibly plot all of them and look at all of them.

## Rhat statistic
The Rhat statistic measures how similar the different chains are. So to be able to compute it, we need at least two chains. If all chains converged to the same distribution, then the Rhat statistic should be close to 1.
An Rhat greater than 1.1 means something is very wrong and your model did not converge (in this case, PyMC3 will also raise some warnings). However, already a value above 1.01 is reason for concern, so when computing it via the summary function from ArviZ, it is best to deactive rounding.


```python
s = az.summary(data, round_to="none")
s.head()
```

I built a small convenience function that plots a histogram of the Rhat over all parameters and also tells you which parameters (if any) are above a certain threshold. This way, we can then analyse these parameters further.


```python
check_rhat(data, threshold=1.05)
plt.show()
```


```python
check_rhat(data, threshold=1.005)
plt.show()
```


```python
az.plot_trace(data, var_names=["alpha", 'beta'], 
              coords={"zip_code": [zip_codes[205], zip_codes[191]]})
plt.show()
```

Compare this with the bad model, a huge bunch of parameters have a very Rhat! Not very good!


```python
check_rhat(bad_data, threshold=1.01)
plt.show()
```

## Number of effective samples
Since the samples coming from one chain are usually autocorrelated. This means, the posterior sample we get is not an independent sample of the posterior distribution. The ESS, Effective sample size, estimates how many independent draws we roughly have in our sample. 
If the number of effective samples is very low (e.g. less than 10%) compared to the number of iterations (size of the posterior sample), then there might be a problem with our model. 

Note that it depends on your use case how many effective samples you need. If you only want to estimate the mean and median of the posterior distribution, then ~300 effective samples can be enough. If however you want to estimate very high or low percentiles, you will need much more.

For a bit more about the maths behind it, check this [section](https://mc-stan.org/docs/2_20/reference-manual/effective-sample-size-section.html).

I build a similr convenience function for the ESS:


```python
check_neff(data, threshold=0.1)
plt.show()
```

## Monte Carlo Standard Error
The Monte Carlo standard error is given by the posterior standard deviation divided by the square root of the number of effective samples. The smaller the standard error, the closer the posterior mean is expected to be to the true value. This standard error should not be greater than 10% of the posterior standard deviation.
For more details, check e.g. the [Stan user manual](https://mc-stan.org/docs/2_18/reference-manual/effective-sample-size-section.html).


```python
check_mcse(data)
plt.show()
```

# Some bad examples
In the next section, I fit a few models that have not converged, just to show how this looks like in the convergence diagnostics.


```python
# Data of the Eight Schools Model
J = 8
y = np.array([28.,  8., -3.,  7., -1.,  1., 18., 12.])
sigma = np.array([15., 10., 16., 11.,  9., 11., 10., 18.])
# tau = 25.
with pm.Model() as Centered_eight:
    mu = pm.Normal('mu', mu=0, sigma=5)
    tau = pm.HalfCauchy('tau', beta=5)
    theta = pm.Normal('theta', mu=mu, sigma=tau, shape=J)
    obs = pm.Normal('obs', mu=theta, sigma=sigma, observed=y)
    posterior = pm.sample(tune=50, draws=500)
```

This model has only been tuned for a very short time and this leads to various convergence problems. PyMC itself warns us that the Rhat (here called Gelman-Rubin statistic) is large and that the number of effective samples is very small. Also there were quite many divergences.

We also see this in the trace plots:


```python
az.plot_trace(posterior, var_names=["mu"])
plt.show()
```

Increasing the tuning steps leads to less divergences and less warnings, but there are still many problems with this model.


```python
# Data of the Eight Schools Model
J = 8
y = np.array([28.,  8., -3.,  7., -1.,  1., 18., 12.])
sigma = np.array([15., 10., 16., 11.,  9., 11., 10., 18.])
# tau = 25.
with pm.Model() as Centered_eight:
    mu = pm.Normal('mu', mu=0, sigma=5)
    tau = pm.HalfCauchy('tau', beta=5)
    theta = pm.Normal('theta', mu=mu, sigma=tau, shape=J)
    obs = pm.Normal('obs', mu=theta, sigma=sigma, observed=y)
    posterior = pm.sample(tune=500, draws=500)
```


```python
az.plot_trace(posterior, var_names=["mu"])
plt.show()
```

We can see in the trace plots that the chains did not explore the posterior space well and that there is high autocorrelation. We can plot the autocorrelation:


```python
axes = az.plot_autocorr(posterior, var_names=["mu"], figsize=(13,6))
for axe in axes:
    for ax in axe:
        ax.spines["top"].set_visible(False)  
        ax.spines["right"].set_visible(False)
plt.show()
```

For comparison, a model with low autocorrelation looks like this:


```python
axes = az.plot_autocorr(data, var_names=["mu_alpha"], figsize=(13,6))
for axe in axes:
    for ax in axe:
        ax.spines["top"].set_visible(False)  
        ax.spines["right"].set_visible(False)
plt.show()
```

The model above is the classic example of where a centered parametrization is problematic. 
We thus try a non-centered parametrization:


```python
with pm.Model() as NonCentered_eight:
    mu = pm.Normal('mu', mu=0, sigma=5)
    tau = pm.HalfCauchy('tau', beta=5)
    theta_tilde = pm.Normal('theta_t', mu=0, sigma=1, shape=J)
    theta = pm.Deterministic('theta', mu + tau * theta_tilde)
    obs = pm.Normal('obs', mu=theta, sigma=sigma, observed=y)
    posterior = pm.sample(target_accept=0.9, draws=500, tune=500)
```


```python
az.plot_trace(posterior, var_names=["mu"])
plt.show()
```

Now, the trace plot looks much better.

The trace plots for the bad model:


```python
az.plot_trace(bad_data, var_names=['alpha', "beta"], coords={"zip_code": [zip_codes[54]],
                                                    "chain": [0,1]})
plt.show()
```
