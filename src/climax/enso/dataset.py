import xarray as xr
import torch
from torch.utils.data import Dataset


class ENSODataset(Dataset):
    def __init__(self, nc_file, lead_time):
        era5_dataset = xr.load_dataset(nc_file)  # TODO: double check number types

        self.era5_daily = torch.from_numpy(era5_dataset.to_array().to_numpy())  # V x T x W x H
        self.lat = era5_dataset.lat.to_numpy()
        self.lon = era5_dataset.lon.to_numpy()

        self.enso_index = None  # TODO

        self.lead_time = lead_time

    def __len__(self):
        return self.era5_daily.shape[1] - self.lead_time

    def __getitem__(self, idx):
        return self.era5_daily[:, idx, :, :], self.enso_index[idx + self.lead_time]
