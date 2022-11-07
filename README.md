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
    mass_1.append(grid_point.intrinsic_table["mass1"][0])
    mass_2.append(grid_point.intrinsic_table["mass2"][0])
    marg_log_likelihood.append(grid_point.intrinsic_table["snr"][0])

# Plot marginalized-log-likelihood over intrinsic parameter (mass_1/mass_2) grid points
plt.scatter(mass_1, mass_2, c=marg_log_likelihood )
plt.xlabel("$m_1$")
plt.ylabel("$m_2$")
plt.colorbar(label="$\ln(L_{marg})$")
```
