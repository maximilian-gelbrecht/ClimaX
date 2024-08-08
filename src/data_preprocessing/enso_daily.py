import glob
import os

import click
import xarray as xr
from tqdm import tqdm


def enso_daily(
    data_dir: str,
    save_dir: str,
    start_year,
    end_year,
    N_days_rolling=30,
    lons=[190, 240],
    lats=[-5, 5],
    normalize: bool = False,
    name_pattern_year_twice: bool = True,
):
    """Computes a daily rolling mean ENSO index directly from netCDF Files and saves it as a .nc file.

    Args:
        data_dir (str): path to the netCDF files
        save_dir (str): path to location the ENSO index will be saved
        years (range): years to compute the index for
        N_days_rolling (int, optional): Length of rolling mean. Defaults to 30.
        lons (list, optional): Longitudes of ENSO box. Defaults to [190, 240].
        lats (list, optional): Latitudes of ENSO box. Defaults to [-5, 5].
        normalize (bool, optional): Normalize the anomalies?. Defaults to False.
        name_pattern_year_twice (bool, optional): In the name pattern for the nc Files the year appears twice, if not once. Defaults to True
    """
    assert start_year < end_year
    assert lons[1] > lons[0]
    assert lats[1] > lats[0]

    os.makedirs(save_dir, exist_ok=True)

    # region Niño 3.4 (5N-5S, 170W-120W):
    min_lon, max_lon = lons
    min_lat, max_lat = lats

    daily_sst_box = []
    for year in tqdm(range(start_year, end_year + 1)):
        # get daily SST in ENSO box from hourly ERA5 SST data
        if name_pattern_year_twice:
            ps = glob.glob(
                os.path.join(data_dir, f"*{year}*{year}*.nc")
            )  # the pattern with just a single 'year' leads to wrong files being loaded with the naming scheme on PIK HPC
        else:
            ps = glob.glob(os.path.join(data_dir, f"*{year}*.nc"))

        dataset_sst = xr.open_mfdataset(ps, combine="by_coords", parallel=True)

        daily_sst_box.append(
            dataset_sst["sst"]
            .sel(latitude=slice(max_lat, min_lat), longitude=slice(min_lon, max_lon))
            .resample(time="1D")
            .mean("time")
            .load()  # Load to avoid out-of-order index warning
        )

    # merge daily_sst_box into one xarray dataset
    daily_sst_box = xr.concat(daily_sst_box, "time")

    # compute anomalies
    gb = daily_sst_box.groupby("time.dayofyear")
    clim = gb.mean(dim="time")
    clim_std = gb.std(dim="time")

    anom = gb - clim

    if normalize:
        anom = anom / clim_std

    # aggregate to daily index
    enso_index_daily = anom.mean(dim=("latitude", "longitude"))

    # rolling mean
    enso_index_rolling = enso_index_daily.rolling(time=N_days_rolling).mean()

    # save
    enso_index_rolling.to_netcdf(os.path.join(save_dir, "enso_daily.nc"))


@click.command()
@click.option(
    "--data-dir", type=click.Path(exists=True), default="/p/projects/climate_data_central/reanalysis/ERA5/sst"
)
@click.option("--save-dir", type=str)
@click.option("--start-year", type=int, default=1979)
@click.option("--end-year", type=int, default=2018)
@click.option("--n-days-rolling", type=int, default=30)
def main(data_dir, save_dir, start_year, end_year, n_days_rolling):
    enso_daily(data_dir, save_dir, start_year, end_year, n_days_rolling)


if __name__ == "__main__":
    main()
