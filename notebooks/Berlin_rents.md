# Berlin Rents
I scraped a data set from Immoscout with rental offers in Berlin. The data set is similar enough to the house price data set, so that we can do a very similar analysis.


```python
import sys
sys.path.append('../src/')

import numpy as np
import pandas as pd
import pymc3 as pm
import arviz as az
import theano
import matplotlib.pyplot as plt

from utils.data_utils import load_data, standardize_area, map_zip_codes
from utils.plot_utils import set_plot_defaults, plot_pred_hist
```


```python
set_plot_defaults(font="Roboto")
d, zip_lookup, num_zip_codes = load_data(kind="rents")
zip_codes = np.sort(d.zip.unique())

target = "rent_s"
```


```python
d
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
      <th>zip</th>
      <th>rent</th>
      <th>rent_s</th>
      <th>log_rent</th>
      <th>log_rent_s</th>
      <th>sqm_rent</th>
      <th>log_sqm_rent</th>
      <th>log_sqm_rent_s</th>
      <th>living_space</th>
      <th>living_space_s</th>
      <th>offer_year</th>
      <th>const_year</th>
      <th>const_year_s</th>
      <th>flattype</th>
      <th>interior_qual</th>
      <th>zip_code</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>59</td>
      <td>12103</td>
      <td>941.00</td>
      <td>9.4100</td>
      <td>6.846943</td>
      <td>-0.037463</td>
      <td>30.354839</td>
      <td>3.412956</td>
      <td>2.350673</td>
      <td>31.00</td>
      <td>-1.145711</td>
      <td>2019</td>
      <td>2019.0</td>
      <td>1.152406</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>84</td>
    </tr>
    <tr>
      <td>75</td>
      <td>14109</td>
      <td>1399.70</td>
      <td>13.9970</td>
      <td>7.244013</td>
      <td>0.651780</td>
      <td>16.999028</td>
      <td>2.833156</td>
      <td>0.728184</td>
      <td>82.34</td>
      <td>-0.002587</td>
      <td>2019</td>
      <td>1920.0</td>
      <td>-0.933787</td>
      <td>apartment</td>
      <td>sophisticated</td>
      <td>199</td>
    </tr>
    <tr>
      <td>84</td>
      <td>13086</td>
      <td>830.00</td>
      <td>8.3000</td>
      <td>6.721426</td>
      <td>-0.255338</td>
      <td>8.251317</td>
      <td>2.110373</td>
      <td>-1.294424</td>
      <td>100.59</td>
      <td>0.403763</td>
      <td>2019</td>
      <td>1912.0</td>
      <td>-1.102368</td>
      <td>ground_floor</td>
      <td>normal</td>
      <td>148</td>
    </tr>
    <tr>
      <td>86</td>
      <td>10785</td>
      <td>3417.60</td>
      <td>34.1760</td>
      <td>8.136694</td>
      <td>2.201314</td>
      <td>16.000000</td>
      <td>2.772589</td>
      <td>0.558695</td>
      <td>213.60</td>
      <td>2.920016</td>
      <td>2019</td>
      <td>2018.0</td>
      <td>1.131333</td>
      <td>penthouse</td>
      <td>sophisticated</td>
      <td>55</td>
    </tr>
    <tr>
      <td>136</td>
      <td>12057</td>
      <td>490.00</td>
      <td>4.9000</td>
      <td>6.194405</td>
      <td>-1.170151</td>
      <td>10.000000</td>
      <td>2.302585</td>
      <td>-0.756545</td>
      <td>49.00</td>
      <td>-0.744927</td>
      <td>2019</td>
      <td>1972.0</td>
      <td>0.161991</td>
      <td>apartment</td>
      <td>normal</td>
      <td>79</td>
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
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>198228</td>
      <td>10997</td>
      <td>1170.00</td>
      <td>11.7000</td>
      <td>7.064759</td>
      <td>0.340627</td>
      <td>18.465909</td>
      <td>2.915926</td>
      <td>0.959805</td>
      <td>63.36</td>
      <td>-0.425191</td>
      <td>2019</td>
      <td>1905.0</td>
      <td>-1.249877</td>
      <td>apartment</td>
      <td>NaN</td>
      <td>69</td>
    </tr>
    <tr>
      <td>198281</td>
      <td>10367</td>
      <td>378.95</td>
      <td>3.7895</td>
      <td>5.937404</td>
      <td>-1.616259</td>
      <td>11.000000</td>
      <td>2.397895</td>
      <td>-0.489833</td>
      <td>34.45</td>
      <td>-1.068894</td>
      <td>2018</td>
      <td>1975.0</td>
      <td>0.225209</td>
      <td>apartment</td>
      <td>NaN</td>
      <td>19</td>
    </tr>
    <tr>
      <td>198298</td>
      <td>12559</td>
      <td>579.00</td>
      <td>5.7900</td>
      <td>6.361302</td>
      <td>-0.880448</td>
      <td>9.650000</td>
      <td>2.266958</td>
      <td>-0.856243</td>
      <td>60.00</td>
      <td>-0.500004</td>
      <td>2019</td>
      <td>1997.0</td>
      <td>0.688807</td>
      <td>apartment</td>
      <td>normal</td>
      <td>129</td>
    </tr>
    <tr>
      <td>198318</td>
      <td>10557</td>
      <td>1748.00</td>
      <td>17.4800</td>
      <td>7.466228</td>
      <td>1.037504</td>
      <td>18.400000</td>
      <td>2.912351</td>
      <td>0.949799</td>
      <td>95.00</td>
      <td>0.279297</td>
      <td>2019</td>
      <td>2019.0</td>
      <td>1.152406</td>
      <td>ground_floor</td>
      <td>sophisticated</td>
      <td>31</td>
    </tr>
    <tr>
      <td>198361</td>
      <td>12167</td>
      <td>1380.00</td>
      <td>13.8000</td>
      <td>7.229839</td>
      <td>0.627176</td>
      <td>18.648649</td>
      <td>2.925774</td>
      <td>0.987362</td>
      <td>74.00</td>
      <td>-0.188283</td>
      <td>2019</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>ground_floor</td>
      <td>sophisticated</td>
      <td>93</td>
    </tr>
  </tbody>
</table>
<p>7258 rows × 16 columns</p>
</div>




```python

```
