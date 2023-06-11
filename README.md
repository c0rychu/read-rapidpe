Read Rapid-PE
===


[![PyPI Version](https://img.shields.io/pypi/v/read-rapidpe?label=PyPI%20version)](
https://pypi.org/project/read-rapidpe/)
[![PyPI Downloads](https://img.shields.io/pypi/dm/read-rapidpe?label=PyPI%20downloads)](
https://pypi.org/project/read-rapidpe/)


This is a package to read Rapid-PE outputs.

# Install
## Install from PyPI
This `read-rapidpe` package is available on PyPI: https://pypi.org/project/read-rapidpe/
```
pip install read-rapidpe
```

## Install in dev mode
```
git clone git@git.ligo.org:yu-kuang.chu/read-rapidpe.git
cd read-rapidpe
pip install -e . 
```

# Example Usage
## Reading files
```python
from read_rapidpe import RapidPE_result

run_dir = "path/to/run_dir"
result = RapidPE_result.from_run_dir(run_dir)
```
There are three optional arguments:
- `use_ligolw` ( default = `True` ) : whether to use `ligo.lw` to read XML files.
- `extrinsic_table` ( default = `True` ) : whether to load extrinsic parameter as well.
- `parallel_n` ( default = `1` ) : number of parallel jobs when reading XML files.

For example, one can do the following to speed up the reading process:
```python
result = RapidPE_result.from_run_dir(run_dir, use_ligolw=False, extrinsic_table=False, parallel_n=4)
```

## Plot marginalized log-likelihood on m1-m2 grid points
```python
import matplotlib.pyplot as plt

# Plot marginalized-log-likelihood over intrinsic parameter (mass_1/mass_2) grid points
plt.scatter(result.mass_1, result.mass_2, c=result.marg_log_likelihood )
plt.xlabel("$m_1$")
plt.ylabel("$m_2$")
plt.colorbar(label="$\ln(L_{marg})$")
```

## Plot interpolated likelihood
```python
import matplotlib.pyplot as plt
import numpy as np


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

