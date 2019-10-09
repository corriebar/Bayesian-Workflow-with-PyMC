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

    WARNING (theano.tensor.blas): Using NumPy C-API based implementation for BLAS functions.
    /home/corrie/.pyenv/versions/anaconda3-2019.03/lib/python3.7/site-packages/xarray/core/merge.py:17: FutureWarning: The Panel class is removed from pandas. Accessing it from the top-level namespace will also be removed in the next version
      PANDAS_TYPES = (pd.Series, pd.DataFrame, pd.Panel)



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




    Inference data with groups:
    	> posterior
    	> sample_stats
    	> posterior_predictive
    	> prior
    	> observed_data
    	> constant_data




```python
bad_data = az.from_netcdf("../models/bad_model.nc")
```


```python
az.summary(data.posterior)
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



## Trace plots
The first thing to check to see if a model converged or not are the trace plots. If you have more than 2 chains, it can be a good idea to only look at two, otherwise the plots can get very crowded.


```python
az.plot_trace(data, var_names=["mu_alpha", "mu_beta", "sigma_alpha", "sigma_beta"],
             coords={"chain":[0,1]})
plt.suptitle("Trace plots")
plt.show()
```


![png](05_Convergency%20Checks_files/05_Convergency%20Checks_8_0.png)


These trace plots look goot but we only look here at a selection of parameters. Remember that for each ZIP code, we fit both an $\alpha$ and $\beta$ parameter.
This means we actually have hundreds of parameter and we cannot feasibly plot all of them and look at all of them.

## Rhat statistic
The Rhat statistic measures how similar the different chains are. So to be able to compute it, we need at least two chains. If all chains converged to the same distribution, then the Rhat statistic should be close to 1.
An Rhat greater than 1.1 means something is very wrong and your model did not converge (in this case, PyMC3 will also raise some warnings). However, already a value above 1.01 is reason for concern, so when computing it via the summary function from ArviZ, it is best to deactive rounding.


```python
s = az.summary(data, round_to="none")
s.head()
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
      <td>3.641892</td>
      <td>0.056519</td>
      <td>3.526990</td>
      <td>3.741161</td>
      <td>0.000760</td>
      <td>0.000538</td>
      <td>5523.662786</td>
      <td>5523.662786</td>
      <td>5534.529199</td>
      <td>3184.033781</td>
      <td>1.000298</td>
    </tr>
    <tr>
      <td>mu_beta</td>
      <td>2.774866</td>
      <td>0.099303</td>
      <td>2.595929</td>
      <td>2.965907</td>
      <td>0.001306</td>
      <td>0.000924</td>
      <td>5782.110792</td>
      <td>5780.776788</td>
      <td>5765.617689</td>
      <td>3343.464957</td>
      <td>1.000784</td>
    </tr>
    <tr>
      <td>alpha[0]</td>
      <td>4.976080</td>
      <td>0.160512</td>
      <td>4.687661</td>
      <td>5.288980</td>
      <td>0.002142</td>
      <td>0.001518</td>
      <td>5617.647832</td>
      <td>5590.325691</td>
      <td>5626.214679</td>
      <td>3249.134036</td>
      <td>1.000693</td>
    </tr>
    <tr>
      <td>alpha[1]</td>
      <td>5.096531</td>
      <td>0.289015</td>
      <td>4.544702</td>
      <td>5.616254</td>
      <td>0.003726</td>
      <td>0.002634</td>
      <td>6018.064316</td>
      <td>6018.064316</td>
      <td>6017.194491</td>
      <td>3308.938699</td>
      <td>1.000847</td>
    </tr>
    <tr>
      <td>alpha[2]</td>
      <td>5.127996</td>
      <td>0.201097</td>
      <td>4.730814</td>
      <td>5.483099</td>
      <td>0.002366</td>
      <td>0.001673</td>
      <td>7223.462402</td>
      <td>7223.462402</td>
      <td>7271.583124</td>
      <td>3517.120722</td>
      <td>0.999774</td>
    </tr>
  </tbody>
</table>
</div>



I built a small convenience function that plots a histogram of the Rhat over all parameters and also tells you which parameters (if any) are above a certain threshold. This way, we can then analyse these parameters further.


```python
check_rhat(data, threshold=1.05)
plt.show()
```

    The following parameters have an Rhat greater 1.05:
    None
    



![png](05_Convergency%20Checks_files/05_Convergency%20Checks_13_1.png)



```python
check_rhat(data, threshold=1.005)
plt.show()
```

    The following parameters have an Rhat greater 1.005:
    ['alpha[205]', 'beta[191]']
    



![png](05_Convergency%20Checks_files/05_Convergency%20Checks_14_1.png)



```python
az.plot_trace(data, var_names=["alpha", 'beta'], 
              coords={"zip_code": [zip_codes[205], zip_codes[191]]})
plt.show()
```


![png](05_Convergency%20Checks_files/05_Convergency%20Checks_15_0.png)


Compare this with the bad model, a huge bunch of parameters have a very Rhat! Not very good!


```python
check_rhat(bad_data, threshold=1.01)
plt.show()
```

    The following parameters have an Rhat greater 1.01:
    ['mu_alpha', 'mu_beta', 'alpha[0]', 'alpha[1]', 'alpha[2]', 'alpha[3]', 'alpha[4]', 'alpha[5]', 'alpha[6]', 'alpha[7]', 'alpha[8]', 'alpha[9]', 'alpha[10]', 'alpha[11]', 'alpha[12]', 'alpha[13]', 'alpha[14]', 'alpha[15]', 'alpha[16]', 'alpha[18]', 'alpha[19]', 'alpha[20]', 'alpha[21]', 'alpha[22]', 'alpha[23]', 'alpha[24]', 'alpha[26]', 'alpha[27]', 'alpha[28]', 'alpha[29]', 'alpha[30]', 'alpha[32]', 'alpha[34]', 'alpha[35]', 'alpha[36]', 'alpha[37]', 'alpha[38]', 'alpha[39]', 'alpha[40]', 'alpha[45]', 'alpha[46]', 'alpha[47]', 'alpha[48]', 'alpha[50]', 'alpha[51]', 'alpha[52]', 'alpha[53]', 'alpha[54]', 'alpha[56]', 'alpha[57]', 'alpha[60]', 'alpha[61]', 'alpha[62]', 'alpha[63]', 'alpha[64]', 'alpha[65]', 'alpha[66]', 'alpha[68]', 'alpha[69]', 'alpha[72]', 'alpha[73]', 'alpha[74]', 'alpha[75]', 'alpha[77]', 'alpha[78]', 'alpha[79]', 'alpha[80]', 'alpha[82]', 'alpha[83]', 'alpha[88]', 'alpha[89]', 'alpha[90]', 'alpha[91]', 'alpha[92]', 'alpha[93]', 'alpha[94]', 'alpha[95]', 'alpha[96]', 'alpha[97]', 'alpha[98]', 'alpha[99]', 'alpha[100]', 'alpha[101]', 'alpha[103]', 'alpha[105]', 'alpha[107]', 'alpha[108]', 'alpha[109]', 'alpha[110]', 'alpha[112]', 'alpha[113]', 'alpha[118]', 'alpha[119]', 'alpha[120]', 'alpha[121]', 'alpha[122]', 'alpha[124]', 'alpha[126]', 'alpha[129]', 'alpha[130]', 'alpha[131]', 'alpha[132]', 'alpha[133]', 'alpha[135]', 'alpha[136]', 'alpha[138]', 'alpha[139]', 'alpha[141]', 'alpha[142]', 'alpha[143]', 'alpha[145]', 'alpha[146]', 'alpha[147]', 'alpha[148]', 'alpha[149]', 'alpha[150]', 'alpha[151]', 'alpha[152]', 'alpha[154]', 'alpha[155]', 'alpha[157]', 'alpha[158]', 'alpha[159]', 'alpha[160]', 'alpha[164]', 'alpha[166]', 'alpha[168]', 'alpha[169]', 'alpha[170]', 'alpha[171]', 'alpha[172]', 'alpha[173]', 'alpha[174]', 'alpha[175]', 'alpha[176]', 'alpha[178]', 'alpha[179]', 'alpha[180]', 'alpha[181]', 'alpha[182]', 'alpha[183]', 'alpha[184]', 'alpha[185]', 'alpha[187]', 'alpha[190]', 'alpha[191]', 'alpha[192]', 'alpha[193]', 'alpha[194]', 'alpha[196]', 'alpha[197]', 'alpha[198]', 'alpha[199]', 'alpha[200]', 'alpha[201]', 'alpha[202]', 'alpha[203]', 'alpha[205]', 'alpha[207]', 'alpha[208]', 'alpha[209]', 'alpha[210]', 'alpha[211]', 'alpha[212]', 'alpha[213]', 'alpha[215]', 'alpha[216]', 'alpha[217]', 'alpha[218]', 'beta[0]', 'beta[1]', 'beta[2]', 'beta[4]', 'beta[5]', 'beta[6]', 'beta[7]', 'beta[8]', 'beta[11]', 'beta[12]', 'beta[13]', 'beta[14]', 'beta[15]', 'beta[16]', 'beta[17]', 'beta[18]', 'beta[19]', 'beta[20]', 'beta[21]', 'beta[22]', 'beta[23]', 'beta[24]', 'beta[25]', 'beta[27]', 'beta[28]', 'beta[29]', 'beta[30]', 'beta[31]', 'beta[34]', 'beta[35]', 'beta[36]', 'beta[37]', 'beta[38]', 'beta[39]', 'beta[41]', 'beta[42]', 'beta[43]', 'beta[44]', 'beta[46]', 'beta[47]', 'beta[48]', 'beta[49]', 'beta[50]', 'beta[53]', 'beta[57]', 'beta[60]', 'beta[61]', 'beta[62]', 'beta[63]', 'beta[64]', 'beta[65]', 'beta[66]', 'beta[67]', 'beta[68]', 'beta[69]', 'beta[72]', 'beta[73]', 'beta[74]', 'beta[76]', 'beta[78]', 'beta[79]', 'beta[80]', 'beta[82]', 'beta[84]', 'beta[86]', 'beta[89]', 'beta[90]', 'beta[91]', 'beta[92]', 'beta[93]', 'beta[94]', 'beta[96]', 'beta[98]', 'beta[99]', 'beta[100]', 'beta[101]', 'beta[102]', 'beta[103]', 'beta[104]', 'beta[105]', 'beta[106]', 'beta[107]', 'beta[108]', 'beta[109]', 'beta[110]', 'beta[111]', 'beta[112]', 'beta[113]', 'beta[115]', 'beta[116]', 'beta[118]', 'beta[121]', 'beta[122]', 'beta[123]', 'beta[124]', 'beta[125]', 'beta[126]', 'beta[127]', 'beta[128]', 'beta[130]', 'beta[131]', 'beta[133]', 'beta[134]', 'beta[136]', 'beta[139]', 'beta[140]', 'beta[141]', 'beta[143]', 'beta[144]', 'beta[146]', 'beta[150]', 'beta[151]', 'beta[152]', 'beta[153]', 'beta[154]', 'beta[155]', 'beta[158]', 'beta[159]', 'beta[160]', 'beta[162]', 'beta[164]', 'beta[165]', 'beta[166]', 'beta[167]', 'beta[168]', 'beta[169]', 'beta[170]', 'beta[171]', 'beta[172]', 'beta[173]', 'beta[174]', 'beta[175]', 'beta[176]', 'beta[180]', 'beta[181]', 'beta[182]', 'beta[183]', 'beta[184]', 'beta[185]', 'beta[186]', 'beta[187]', 'beta[188]', 'beta[189]', 'beta[191]', 'beta[192]', 'beta[193]', 'beta[194]', 'beta[196]', 'beta[198]', 'beta[200]', 'beta[201]', 'beta[202]', 'beta[203]', 'beta[204]', 'beta[205]', 'beta[206]', 'beta[207]', 'beta[209]', 'beta[210]', 'beta[211]', 'beta[212]', 'beta[213]', 'beta[214]', 'beta[215]', 'beta[216]', 'sigma_alpha', 'sigma_beta', 'sigma[0]', 'sigma[1]', 'sigma[2]', 'sigma[3]', 'sigma[4]', 'sigma[5]', 'sigma[6]', 'sigma[7]', 'sigma[8]', 'sigma[9]', 'sigma[10]', 'sigma[12]', 'sigma[13]', 'sigma[14]', 'sigma[15]', 'sigma[16]', 'sigma[18]', 'sigma[19]', 'sigma[20]', 'sigma[22]', 'sigma[23]', 'sigma[24]', 'sigma[26]', 'sigma[27]', 'sigma[28]', 'sigma[29]', 'sigma[30]', 'sigma[31]', 'sigma[32]', 'sigma[33]', 'sigma[35]', 'sigma[36]', 'sigma[37]', 'sigma[39]', 'sigma[40]', 'sigma[41]', 'sigma[42]', 'sigma[45]', 'sigma[46]', 'sigma[47]', 'sigma[48]', 'sigma[49]', 'sigma[50]', 'sigma[51]', 'sigma[52]', 'sigma[54]', 'sigma[55]', 'sigma[57]', 'sigma[60]', 'sigma[61]', 'sigma[62]', 'sigma[63]', 'sigma[65]', 'sigma[66]', 'sigma[67]', 'sigma[68]', 'sigma[69]', 'sigma[72]', 'sigma[73]', 'sigma[74]', 'sigma[75]', 'sigma[77]', 'sigma[78]', 'sigma[79]', 'sigma[81]', 'sigma[82]', 'sigma[83]', 'sigma[84]', 'sigma[85]', 'sigma[86]', 'sigma[87]', 'sigma[88]', 'sigma[89]', 'sigma[90]', 'sigma[91]', 'sigma[92]', 'sigma[93]', 'sigma[94]', 'sigma[95]', 'sigma[96]', 'sigma[98]', 'sigma[99]', 'sigma[100]', 'sigma[101]', 'sigma[102]', 'sigma[104]', 'sigma[105]', 'sigma[106]', 'sigma[107]', 'sigma[108]', 'sigma[110]', 'sigma[112]', 'sigma[113]', 'sigma[114]', 'sigma[115]', 'sigma[117]', 'sigma[118]', 'sigma[119]', 'sigma[121]', 'sigma[122]', 'sigma[123]', 'sigma[125]', 'sigma[126]', 'sigma[127]', 'sigma[128]', 'sigma[129]', 'sigma[130]', 'sigma[132]', 'sigma[134]', 'sigma[135]', 'sigma[136]', 'sigma[137]', 'sigma[138]', 'sigma[139]', 'sigma[140]', 'sigma[142]', 'sigma[143]', 'sigma[144]', 'sigma[145]', 'sigma[149]', 'sigma[150]', 'sigma[152]', 'sigma[153]', 'sigma[154]', 'sigma[155]', 'sigma[157]', 'sigma[158]', 'sigma[159]', 'sigma[160]', 'sigma[162]', 'sigma[163]', 'sigma[164]', 'sigma[165]', 'sigma[166]', 'sigma[167]', 'sigma[168]', 'sigma[169]', 'sigma[170]', 'sigma[172]', 'sigma[173]', 'sigma[175]', 'sigma[176]', 'sigma[178]', 'sigma[179]', 'sigma[181]', 'sigma[183]', 'sigma[184]', 'sigma[186]', 'sigma[187]', 'sigma[188]', 'sigma[189]', 'sigma[190]', 'sigma[191]', 'sigma[192]', 'sigma[195]', 'sigma[197]', 'sigma[198]', 'sigma[199]', 'sigma[200]', 'sigma[202]', 'sigma[203]', 'sigma[205]', 'sigma[206]', 'sigma[208]', 'sigma[209]', 'sigma[210]', 'sigma[211]', 'sigma[212]', 'sigma[213]', 'sigma[214]', 'sigma[216]', 'sigma[217]', 'sigma[218]']
    



![png](05_Convergency%20Checks_files/05_Convergency%20Checks_17_1.png)


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

    The following parameters have an effective sample size less than 10.0% of the total sample size:
    None
    



![png](05_Convergency%20Checks_files/05_Convergency%20Checks_19_1.png)


## Monte Carlo Standard Error
The Monte Carlo standard error is given by the posterior standard deviation divided by the square root of the number of effective samples. The smaller the standard error, the closer the posterior mean is expected to be to the true value. This standard error should not be greater than 10% of the posterior standard deviation.
For more details, check e.g. the [Stan user manual](https://mc-stan.org/docs/2_18/reference-manual/effective-sample-size-section.html).


```python
check_mcse(data)
plt.show()
```

    The following parameters have a Monte Carlo standard error greater than 10.0% of the posterior standard deviation:
    None
    



![png](05_Convergency%20Checks_files/05_Convergency%20Checks_21_1.png)


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

    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (2 chains in 2 jobs)
    NUTS: [theta, tau, mu]
    Sampling 2 chains: 100%|██████████| 1100/1100 [00:03<00:00, 292.21draws/s]
    There were 6 divergences after tuning. Increase `target_accept` or reparameterize.
    There were 2 divergences after tuning. Increase `target_accept` or reparameterize.
    The acceptance probability does not match the target. It is 0.9612063297357322, but should be close to 0.8. Try to increase the number of tuning steps.
    The estimated number of effective samples is smaller than 200 for some parameters.


This model has only been tuned for a very short time and this leads to various convergence problems. PyMC itself warns us that the Rhat (here called Gelman-Rubin statistic) is large and that the number of effective samples is very small. Also there were quite many divergences.

We also see this in the trace plots:


```python
az.plot_trace(posterior, var_names=["mu"])
plt.show()
```


![png](05_Convergency%20Checks_files/05_Convergency%20Checks_25_0.png)


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

    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (2 chains in 2 jobs)
    NUTS: [theta, tau, mu]
    Sampling 2 chains: 100%|██████████| 2000/2000 [00:03<00:00, 662.92draws/s]
    There were 16 divergences after tuning. Increase `target_accept` or reparameterize.
    There were 52 divergences after tuning. Increase `target_accept` or reparameterize.
    The acceptance probability does not match the target. It is 0.39444060239710466, but should be close to 0.8. Try to increase the number of tuning steps.
    The gelman-rubin statistic is larger than 1.05 for some parameters. This indicates slight problems during sampling.
    The estimated number of effective samples is smaller than 200 for some parameters.



```python
az.plot_trace(posterior, var_names=["mu"])
plt.show()
```


![png](05_Convergency%20Checks_files/05_Convergency%20Checks_28_0.png)


We can see in the trace plots that the chains did not explore the posterior space well and that there is high autocorrelation. We can plot the autocorrelation:


```python
axes = az.plot_autocorr(posterior, var_names=["mu"], figsize=(13,6))
for axe in axes:
    for ax in axe:
        ax.spines["top"].set_visible(False)  
        ax.spines["right"].set_visible(False)
plt.show()
```


![png](05_Convergency%20Checks_files/05_Convergency%20Checks_30_0.png)


For comparison, a model with low autocorrelation looks like this:


```python
axes = az.plot_autocorr(data, var_names=["mu_alpha"], figsize=(13,6))
for axe in axes:
    for ax in axe:
        ax.spines["top"].set_visible(False)  
        ax.spines["right"].set_visible(False)
plt.show()
```


![png](05_Convergency%20Checks_files/05_Convergency%20Checks_32_0.png)


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

    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (2 chains in 2 jobs)
    NUTS: [theta_t, tau, mu]
    Sampling 2 chains: 100%|██████████| 2000/2000 [00:01<00:00, 1140.77draws/s]



```python
az.plot_trace(posterior, var_names=["mu"])
plt.show()
```


![png](05_Convergency%20Checks_files/05_Convergency%20Checks_35_0.png)


Now, the trace plot looks much better.

The trace plots for the bad model:


```python
az.plot_trace(bad_data, var_names=['alpha', "beta"], coords={"zip_code": [zip_codes[54]],
                                                    "chain": [0,1]})
plt.show()
```


![png](05_Convergency%20Checks_files/05_Convergency%20Checks_37_0.png)

