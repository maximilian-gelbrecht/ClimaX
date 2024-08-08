import torch
from torch.utils.data import Dataset


class ENSODataset(Dataset):
    def __init__(self, era5_daily, enso_index, lat, lon):
        self.era5 = torch.from_numpy(era5_daily)  # V x T x W x H
        self.enso = torch.from_numpy(enso_index)
        self.lat = torch.from_numpy(lat)
        self.lon = torch.from_numpy(lon)

    def __len__(self):
        return len(self.enso)

    def __getitem__(self, idx):
        return self.era5[:, idx, :, :], self.enso[idx]
