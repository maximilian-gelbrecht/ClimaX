import glob
import os

import click
import numpy as np
import xarray as xr
from tqdm import tqdm

def nc2np_enso_daily(path: str, save_dir: str, years, N_days_rolling=30, lons=[190, 240], lats=[-5, 5], normalize: bool=False, name_pattern_year_twice: bool=True):
    """Computes a daily rolling mean ENSO index directly from netCDF Files and saves it as .npz files.

    Args:
        path (str): path to the netCDF files 
        save_dir (str): path to location the ENSO index will be saved
        years (range): years to compute the index for
        N_days_rolling (int, optional): Length of rolling mean. Defaults to 30.
        lons (list, optional): Longitudes of ENSO box. Defaults to [190, 240].
        lats (list, optional): Latitudes of ENSO box. Defaults to [-5, 5].
        normalize (bool, optional): Normalize the anomalies?. Defaults to False.
        name_pattern_year_twice (bool, optional): In the name pattern for the nc Files the year appears twice, if not once. Defaults to True
    """
    assert lons[1] > lons[0] 
    assert lats[1] > lats[0]

    # region Niño 3.4 (5N-5S, 170W-120W):
    min_lon = lons[0]
    min_lat = lats[0]
    max_lon = lons[1]
    max_lat = lats[1]

    daily_sst_box = {}

    for year in tqdm(years):

        # get daily SST in ENSO box from hourly ERA5 SST data
        if name_pattern_year_twice:
            ps = glob.glob(os.path.join(path, f"*{year}*{year}*.nc")) # the pattern with just a single 'year' leads to wrong files being loaded with the naming scheme on PIK HPC
        else:
            ps = glob.glob(os.path.join(path, f"*{year}*.nc"))

        ds_sst = xr.open_mfdataset(ps, combine="by_coords", parallel=True)
        
        cropped_ds_sst = ds_sst['sst'].sel(latitude=slice(max_lat,min_lat), longitude=slice(min_lon,max_lon))
        daily_sst_box[year] = cropped_ds_sst.resample(time='1D').mean('time')

    # merge daily_sst_box into one xarray ds and load it into RAM
    daily_sst_box = xr.concat(daily_sst_box.values(), 'time').load()
    
    # compute anomalies 
    gb = daily_sst_box.groupby('time.dayofyear')
    clim = gb.mean(dim='time')
    clim_std = gb.std(dim='time')

    anom = gb - clim 

    if normalize:
        anom = anom / clim_std
    
    # aggregate to daily index 
    enso_index_daily = anom.mean(dim=('latitude', 'longitude'))

    # rolling mean
    enso_index_rolling = enso_index_daily.rolling(time=N_days_rolling).mean()

    # save 
    np.save(os.path.join(save_dir,"enso-rolling-index.npy"), enso_index_rolling.to_numpy())
    np.save(os.path.join(save_dir,"enso-rolling-time.npy"), enso_index_rolling['time'].to_numpy())

@click.command()
@click.option("--root_dir", type=click.Path(exists=True), default='/p/projects/climate_data_central/reanalysis/ERA5/sst')
@click.option("--save_dir", type=str)
@click.option("--start_year", type=int, default=1979)
@click.option("--end_year", type=int, default=2018)
@click.option("--n_days_rolling", type=int, default=30)
def main(
    root_dir,
    save_dir,
    start_year,
    end_year,
    n_days_rolling    
):
    
    assert start_year < end_year 
    os.makedirs(save_dir, exist_ok=True)
    nc2np_enso_daily(root_dir, save_dir, range(start_year, end_year), n_days_rolling)

if __name__ == "__main__":
    main()