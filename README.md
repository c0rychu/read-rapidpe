Read Rapid-PE
===

This is a package to read Rapid-PE outputs.

# Install (dev mode)
```
git clone git@git.ligo.org:yu-kuang.chu/read-rapidpe.git
cd read-rapidpe
pip install -e . 
```

# Example Usage

## Plot marginalized log-likelihood on m1-m2 grid points
```python
from read_rapidpe import RapidPE_result
import matplotlib.pyplot as plt
import glob


results_dir = "path/to/results"
result_xml_files = glob.glob(results_dir+"*.xml.gz")
result = RapidPE_result.from_xml_array(result_xml_files)


# Plot marginalized-log-likelihood over intrinsic parameter (mass_1/mass_2) grid points
plt.scatter(result.mass_1, result.mass_2, c=result.marg_log_likelihood )
plt.xlabel("$m_1$")
plt.ylabel("$m_2$")
plt.colorbar(label="$\ln(L_{marg})$")
```

## Plot interpolated likelihood
```python
from read_rapidpe import RapidPE_result
import matplotlib.pyplot as plt
import glob
import numpy as np


results_dir = "path/to/results"
result_xml_files = glob.glob(results_dir+"*.xml.gz")
result = RapidPE_result.from_xml_array(result_xml_files)


# Create Random m1, m2 samples
m1 = np.random.random(10000)*5
m2 = np.random.random(10000)*5


# After calling result.do_interpolate_marg_log_likelihood_m1m2(), 
# the method result.log_likelihood(m1, m2) will be avalible.
result.do_interpolate_marg_log_likelihood_m1m2()

# Calculate interpolated log_likelihood
log_likelihood = result.log_likelihood(m1, m2)


# =============== Plotting ===============
# Plot interpolated likelihood 
plt.scatter(m1, m2, c=np.exp(log_likelihood), marker=".", s=3, alpha=0.1)

# Plot marginalized likelihood on grid points
plt.scatter(result.mass_1, result.mass_2, c=np.exp(result.marg_log_likelihood), marker="+", vmin=0)

plt.xlabel("$m_1$")
plt.ylabel("$m_2$")
plt.colorbar(label=r"$\mathcal{L}$")
```


## Convert to Pandas DataFrame

```python
import pandas as pd
from read_rapidpe import RapidPE_grid_point


grid_point = RapidPE_grid_point.from_xml("ILE_iteration_xxxxxxxxxx.samples.xml.gz")
pd.DataFrame(grid_point.extrinsic_table)
```

