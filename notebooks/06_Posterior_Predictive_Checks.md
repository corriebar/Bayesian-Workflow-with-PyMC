```python

```


```python
import sys
sys.path.append('../src/')

import numpy as np
import pandas as pd
import pymc3 as pm
import arviz as az
import seaborn as sns

import matplotlib.pyplot as plt
from utils.data_utils import load_data
from utils.plot_utils import set_plot_defaults
from utils.ppc_utils import plot_ppc
```

    /home/corrie/.pyenv/versions/anaconda3-2019.03/lib/python3.7/site-packages/xarray/core/merge.py:17: FutureWarning: The Panel class is removed from pandas. Accessing it from the top-level namespace will also be removed in the next version
      PANDAS_TYPES = (pd.Series, pd.DataFrame, pd.Panel)
    /home/corrie/.pyenv/versions/anaconda3-2019.03/lib/python3.7/site-packages/distributed/utils.py:134: RuntimeWarning: Couldn't detect a suitable IP address for reaching '8.8.8.8', defaulting to '127.0.0.1': [Errno 101] Network is unreachable
      % (host, default, e), RuntimeWarning)



```python
set_plot_defaults("Europace Sans")
d, _, _ = load_data()
```


```python
inf_data = az.from_netcdf("../models/centered_hier.nc")
inf_data
```




    Inference data with groups:
    	> posterior
    	> sample_stats
    	> posterior_predictive
    	> prior
    	> observed_data




```python
ax = az.plot_ppc(inf_data, kind="density", num_pp_samples=100)
ax[0].spines["top"].set_visible(False)  
ax[0].spines["right"].set_visible(False)
ax[0].spines["left"].set_visible(False)
ax[0].set_xlim(-5, 15)
ax[0].legend(frameon=False,  markerscale=3., loc="upper right")
plt.show()
```


![png](06_Posterior_Predictive_Checks_files/06_Posterior_Predictive_Checks_4_0.png)



```python
plt.rcParams['ytick.labelsize'] = 22
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['legend.fontsize'] = 22
plt.rcParams['font.size'] = 20
```


```python
plot_ppc(inf_data, kind="hist")
plt.show()
```


![png](06_Posterior_Predictive_Checks_files/06_Posterior_Predictive_Checks_6_0.png)



```python
fig, ax = plot_ppc(inf_data, kind="density", n=50)
plt.show()
```


![png](06_Posterior_Predictive_Checks_files/06_Posterior_Predictive_Checks_7_0.png)



```python
plot_ppc(inf_data, kind="scatter")
plt.show()
```


![png](06_Posterior_Predictive_Checks_files/06_Posterior_Predictive_Checks_8_0.png)

