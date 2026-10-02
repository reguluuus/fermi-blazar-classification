# Data

This repository does not redistribute the Fermi-LAT catalog.

Download the **4FGL-DR3 / 12-year point-source catalog** (`gll_psc_v31.fit`) from the official Fermi Science Support Center and place it here:

```text
data/gll_psc_v31.fit
```

Official catalog page:
https://fermi.gsfc.nasa.gov/ssc/data/access/lat/12yr_catalog/

The training script reads the main FITS table, keeps BLL, FSRQ and BCU sources, and uses numeric scalar columns as model features. BLL and FSRQ objects form the supervised dataset; BCU objects are reserved for inference.
