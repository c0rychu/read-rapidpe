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

## Convert to Pandas DataFrame

```python
import pandas as pd
from read_rapidpe import RapidPE_grid_point

grid_point = RapidPE_grid_point.from_xml("ILE_iteration_xxxxxxxxxx.samples.xml.gz")
pd.DataFrame(grid_point.extrinsic_table)
```

