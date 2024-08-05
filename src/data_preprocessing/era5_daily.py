import glob
import os

import click
import numpy as np
import xarray as xr
from tqdm import tqdm

from climax.utils.data_utils import DEFAULT_PRESSURE_LEVELS, NAME_TO_VAR


def era5_daily(data_dir, save_dir, variables, start_year, end_year, aggregation_mode):
    assert end_year > start_year
    os.makedirs(os.path.join(save_dir, "era5_daily"), exist_ok=True)

    era5_daily = xr.Dataset()
    for year in tqdm(range(start_year, end_year + 1)):
        for var in variables:
            ps = glob.glob(os.path.join(data_dir, var, f"*{year}*.nc"))
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

    # # Normalization
    # if partition == "train":
    #     era5_daily.mean(dim="time").to_netcdf(os.path.join(save_dir, "era5_daily", "normalize_mean.nc"))
    #     era5_daily.std(dim="time").to_netcdf(os.path.join(save_dir, "era5_daily", "normalize_std.nc"))

    # Add constants
    constants = xr.open_mfdataset(
        os.path.join(data_dir, "constants/constants_5.625deg.nc"), combine="by_coords", parallel=True
    )
    for constant_field in ["land_sea_mask", "orography", "lattitude"]:
        code = NAME_TO_VAR[constant_field]
        era5_daily[code] = constants[code].expand_dims(dim={"time": era5_daily.time})

    era5_daily.to_netcdf(os.path.join(save_dir, "era5_daily.nc"))


@click.command()
@click.option("--data-dir", type=click.Path(exists=True))
@click.option("--save-dir", type=str)
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
@click.option("--start-year", type=int, default=1979)
@click.option("--end-year", type=int, default=2019)
@click.option("--aggregation", type=str, default="mean")
def main(
    data_dir,
    save_dir,
    variables,
    start_year,
    end_year,
    aggregation,
):
    era5_daily(data_dir, save_dir, variables, start_year, end_year, aggregation)


if __name__ == "__main__":
    main()
