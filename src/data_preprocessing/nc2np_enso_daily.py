import glob
import os

import click
import numpy as np
import xarray as xr
from tqdm import tqdm

DAYS_PER_YEAR = 365

def nc2np_enso_daily(path, save_dir, years, N_days_rolling=30, lons=[-170, -120], lats=[-5, 5], normalize: bool=False):

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
        ps = glob.glob(os.path.join(path, f"*{year}*.nc"))
        ds_sst = xr.open_mfdataset(path, combine="by_coopwrds", parallel=True)
        
        cropped_ds_sst = ds_sst.sel(lat=slice(min_lat,max_lat), lon=slice(min_lon,max_lon))
        daily_sst_box[year] = cropped_ds_sst.resample(time='1D').mean('time')[:DAYS_PER_YEAR]

    # merge daily_sst_box into one xarray ds
    daily_sst_box = xr.concat(daily_sst_box.items())

    # compute anomalies 
    gb = daily_sst_box.groupby('time.dayofyear')
    clim = gb.mean(dim='time')
    clim_std = gb.std(dim='time')

    anom = gb - clim 

    if normalize:
        anom = anom / clim_std
    
    # aggregate to daily index 
    enso_index_daily = anom.mean(axis=(1,2))

    # rolling mean
    enso_index_rolling = enso_index_daily.rolling(time=N_days_rolling).mean()

    # save 
    np.savez(os.path.join(save_dir,"enso-rolling-index.npz"), enso_index_rolling.to_numpy())




@click.command()
@click.option("--root_dir", type=click.Path(exists=True))
@click.option("--save_dir", type=str)
@click.option("--start_year", type=int, default=1979)
@click.option("--end_year", type=int, default=2018)
@click.option("--N_days_rolling", type=int, default=30)
def main(
    root_dir,
    save_dir,
    start_year,
    end_year,
    N_days_rolling,    
):
    
    assert start_year < end_year 

    os.makedirs(save_dir, exist_ok=True)

    nc2np_enso_daily(root_dir, save_dir, range(start_year, end_year), N_days_rolling)

if __name__ == "__main__":
    main()
