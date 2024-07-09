# compare the computed running mean enso index to the index downloaded from Copernicus 

import matplotlib.pyplot as plt  # not part of the main environment of ClimaX
import numpy as np 
import xarray as xr 

# download from https://data.marine.copernicus.eu/product/GLOBAL_OMI_CLIMVAR_enso_sst_area_averaged_anomalies/services
enso_reference_dataset = 'global_omi_climate-variability_nino34_sst_anom_19930115_P20220427_R19932014.nc'

running_index = np.load('data/enso-rolling-index.npy')
running_time = np.load('data/enso-rolling-time.npy')

running_enso_ds = xr.DataArray(running_index, dims=["time"], coords=dict(time=running_time))
enso_reference_ds = xr.open_dataset(enso_reference_dataset)

running_enso_ds.plot()
enso_reference_ds['sst_mean'].plot()
plt.show()