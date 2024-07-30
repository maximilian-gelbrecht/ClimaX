import glob
import os

import click
import numpy as np
import xarray as xr
from tqdm import tqdm

from climax.utils.data_utils import DEFAULT_PRESSURE_LEVELS, NAME_TO_VAR


def era5_daily(path, variables, years, save_dir, partition, aggregation_mode):
    os.makedirs(os.path.join(save_dir, partition), exist_ok=True)

    era5_daily = xr.Dataset()
    for year in tqdm(years):
        for var in variables:
            ps = glob.glob(os.path.join(path, var, f"*{year}*.nc"))
            dataset = xr.open_mfdataset(
                ps, combine="by_coords", parallel=True
            )  # dataset for a single variable and year
            code = NAME_TO_VAR[var]

            if len(dataset[code].shape) == 3:  # surface level variables
                # aggregate to daily data, either by just taking one snapshot or by taking a mean
                if aggregation_mode == "mean":
                    resampled = dataset[code].resample(time="1D").mean("time")
                elif aggregation_mode == "snapshot":
                    resampled = dataset[code][::24]

                if code not in era5_daily:
                    era5_daily[code] = resampled
                else:
                    era5_daily = era5_daily.merge(xr.Dataset({code: resampled}))

            else:  # multiple-level variables, only use a subset
                assert len(dataset[code].shape) == 4
                all_levels = dataset["level"][:].to_numpy()
                all_levels = np.intersect1d(all_levels, DEFAULT_PRESSURE_LEVELS)

                for level in all_levels:
                    ds_level = dataset.sel(level=[level]).squeeze(drop=True)

                    # aggregate to daily data, either by just taking one snapshot or by taking a mean
                    var_name = f"{code}_{level}"

                    if aggregation_mode == "mean":
                        resampled = ds_level[code].resample(time="1D").mean("time")
                    elif aggregation_mode == "snapshot":
                        resampled = ds_level[code][::24]

                    if var_name not in era5_daily:
                        era5_daily[var_name] = resampled
                    else:
                        era5_daily = era5_daily.merge(xr.Dataset({var_name: resampled}))

    # Add constants
    constants = xr.open_mfdataset(
        os.path.join(path, "constants/constants_5.625deg.nc"), combine="by_coords", parallel=True
    )
    for constant_field in ["land_sea_mask", "orography", "lattitude"]:
        code = NAME_TO_VAR[constant_field]
        era5_daily[code] = constants[code].expand_dims(dim={"time": era5_daily.time})


@click.command()
@click.option("--root_dir", type=click.Path(exists=True))
@click.option("--save_dir", type=str)
@click.option(
    "--variables",
    "-v",
    type=click.STRING,
    multiple=True,
    default=[
        "2m_temperature",
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "toa_incident_solar_radiation",
        "total_precipitation",
        "geopotential",
        "u_component_of_wind",
        "v_component_of_wind",
        "temperature",
        "relative_humidity",
        "specific_humidity",
    ],
)
@click.option("--start_train_year", type=int, default=1979)
@click.option("--start_val_year", type=int, default=2016)
@click.option("--start_test_year", type=int, default=2017)
@click.option("--end_year", type=int, default=2019)
@click.option("--aggregation", type=str, default="mean")
def main(
    root_dir,
    save_dir,
    variables,
    start_train_year,
    start_val_year,
    start_test_year,
    end_year,
    aggregation,
):
    assert start_val_year > start_train_year and start_test_year > start_val_year and end_year > start_test_year
    train_years = range(start_train_year, start_val_year)
    val_years = range(start_val_year, start_test_year)
    test_years = range(start_test_year, end_year)

    os.makedirs(save_dir, exist_ok=True)

    era5_daily(root_dir, variables, train_years, save_dir, "train", aggregation)
    era5_daily(root_dir, variables, val_years, save_dir, "val", aggregation)
    era5_daily(root_dir, variables, test_years, save_dir, "test", aggregation)

    # save lat and lon data
    ps = glob.glob(os.path.join(root_dir, variables[0], f"*{train_years[0]}*.nc"))
    x = xr.open_mfdataset(ps[0], parallel=True)
    lat = x["lat"].to_numpy()
    lon = x["lon"].to_numpy()
    np.save(os.path.join(save_dir, "lat.npy"), lat)
    np.save(os.path.join(save_dir, "lon.npy"), lon)


if __name__ == "__main__":
    main()
