from datetime import date, timedelta

from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
import xarray as xr

from climax.enso.dataset import ENSODataset


class ENSODataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        start_train_year: int,
        start_val_year: int,
        start_test_year: int,
        end_year: int,
        forecast_lead_time: int,
        batch_size: int,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        # Dataset args
        self.data_dir = data_dir
        self.start_train_year = start_train_year
        self.start_val_year = start_val_year
        self.start_test_year = start_test_year
        self.end_year = end_year
        self.forecast_lead_time = forecast_lead_time

        # Dataloader args
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        # Set up datasets
        # TODO: is here the best place to do this?
        # TODO: fix paths
        era5_dataset = xr.load_dataset(self.data_dir + "/era5_daily.nc")
        enso_dataset = xr.load_dataset(self.data_dir + "/enso_daily.nc")
        self.train_dataset = self.get_dataset(
            era5_dataset,
            enso_dataset,
            self.start_train_year,
            self.start_val_year - 1,
        )
        self.val_dataset = self.get_dataset(
            era5_dataset,
            enso_dataset,
            self.start_val_year,
            self.start_test_year - 1,
        )
        self.test_dataset = self.get_dataset(
            era5_dataset,
            enso_dataset,
            self.start_test_year,
            self.end_year,
        )

    def setup(self, stage):
        pass

    def get_dataset(self, era5_dataset, enso_dataset, start_year, end_year):
        start_date = date(start_year, 1, 1)
        end_date = date(end_year, 12, 31)
        leadtime = timedelta(
            days=self.forecast_lead_time
        )  # TODO: throw error if start_date - leadtime is before the first date in the dataset
        era5 = era5_dataset.sel(time=slice(start_date - leadtime, end_date - leadtime)).to_array().to_numpy()
        enso = enso_dataset.sel(time=slice(start_date, end_date)).to_array().squeeze().to_numpy()
        lat = era5_dataset.lat.to_numpy()
        lon = era5_dataset.lon.to_numpy()
        return ENSODataset(era5, enso, lat, lon)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
