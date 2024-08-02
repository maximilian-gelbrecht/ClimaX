from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule

from climax.enso.dataset import ENSODataset


class ENSODataModule(LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int, lead_time: int):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.lead_time = lead_time

    def setup(self, stage):
        self.enso_train = ENSODataset("path/to/train", self.lead_time)
        self.enso_val = ENSODataset("path/to/val", self.lead_time)
        self.enso_test = ENSODataset("path/to/test", self.lead_time)

    def train_dataloader(self):
        return DataLoader(self.enso_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.enso_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.enso_test, batch_size=self.batch_size)
