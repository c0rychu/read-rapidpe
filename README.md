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

## Plot m1-m2
```python
from read_rapidpe.parser import RapidPE_XML 
import matplotlib.pyplot as plt
import glob

results_dir = "path/to/results"

mass_1 = []
mass_2 = []
marg_log_likelihood = []
for file in glob.glob(results_dir+"*0.xml.gz"):
    grid_point = RapidPE_XML(file)
    mass_1.append(grid_point.intrinsic_table["mass_1"][0])
    mass_2.append(grid_point.intrinsic_table["mass_2"][0])
    marg_log_likelihood.append(grid_point.intrinsic_table["marg_ln_likelihood"][0])

# Plot marginalized-log-likelihood over intrinsic parameter (mass_1/mass_2) grid points
plt.scatter(mass_1, mass_2, c=marg_log_likelihood )
plt.xlabel("$m_1$")
plt.ylabel("$m_2$")
plt.colorbar(label="$\ln(L_{marg})$")
```

## Convert to Pandas DataFrame

```python
import pandas as pd

grid_point = RapidPE_XML("ILE_iteration_xxxxxxxxxx.samples.xml.gz")
pd.DataFrame(grid_point.extrinsic_table)
```