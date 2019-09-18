

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import altair as alt
alt.data_transformers.disable_max_rows()
```




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
      <th>sqm_price</th>
      <th>log_sqm_price</th>
      <th>living_space</th>
      <th>sale_year</th>
      <th>sale_month</th>
      <th>const_year</th>
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
      <td>2246.069379</td>
      <td>7.716937</td>
      <td>80.14</td>
      <td>2018</td>
      <td>8</td>
      <td>1936.0</td>
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
      <td>1740.000000</td>
      <td>7.461640</td>
      <td>50.00</td>
      <td>2017</td>
      <td>7</td>
      <td>1910.0</td>
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
      <td>3066.666667</td>
      <td>8.028346</td>
      <td>33.00</td>
      <td>2016</td>
      <td>4</td>
      <td>2016.0</td>
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
      <td>3420.142890</td>
      <td>8.137438</td>
      <td>43.39</td>
      <td>2017</td>
      <td>1</td>
      <td>2017.0</td>
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
      <td>3374.677003</td>
      <td>8.124055</td>
      <td>77.40</td>
      <td>2016</td>
      <td>6</td>
      <td>2016.0</td>
      <td>flat</td>
      <td>NaN</td>
      <td>rent_out</td>
      <td>sophisticated</td>
    </tr>
  </tbody>
</table>
</div>




```python
alt.Chart(d).mark_bar().encode(
    alt.X("sqm_price:Q", 
            bin=alt.BinParams(maxbins=100)),
    y="count()"
)
```




![png](Berlin_prices_files/Berlin_prices_3_0.png)




```python

```
