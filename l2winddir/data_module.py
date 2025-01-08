import torch
from torch.utils.data import DataLoader, TensorDataset
import xarray as xr
import lightning as L
import numpy as np


class WindDirectionDataModule(L.LightningDataModule):
    """
    DataModule for loading and processing wind direction data for training, validation, and testing.

    Args:
        train_data_paths (str or list of str, optional): Path(s) to the training data files.
        valid_data_paths (str or list of str, optional): Path(s) to the validation data files.
        test_data_paths (str or list of str, optional): Path(s) to the test data files.
        inc (int): Number of input channels (1 for single polarization, 2 for dual polarization).
        batch_size (int): Batch size for the DataLoader.

    Attributes:
        test_dataset (TensorDataset): Dataset for testing.
        valid_dataset (TensorDataset): Dataset for validation.
        train_dataset (TensorDataset): Dataset for training.
        train_mean (torch.Tensor): Mean of the training data (used for normalization).
        train_std (torch.Tensor): Standard deviation of the training data (used for normalization).
    """

    def __init__(self, train_data_paths=None, valid_data_paths=None, test_data_paths=None, inc=2, batch_size=512, pol='VV'):
        super().__init__()
        self.train_data_paths = train_data_paths
        self.valid_data_paths = valid_data_paths
        self.test_data_paths = test_data_paths
        self.inc = inc
        self.batch_size = batch_size
        self.pol = pol
        self.train_mean = None
        self.train_std = None
        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        """
        Placeholder method for any data preparation logic, such as downloading data.
        Not implemented in this module.
        """
        pass

    def setup(self, stage=None):
        """
        Setup method to load and process datasets for different stages of model training.

        Args:
            stage (str, optional): The stage of processing (e.g., 'fit', 'test', 'predict').
        """

        if self.test_data_paths:
            self.test_dataset = self.load_dataset(self.test_data_paths)

        if stage == 'fit' or stage is None:
            if self.train_data_paths:
                self.train_dataset = self.load_dataset(self.train_data_paths)
            if self.valid_data_paths:
                self.valid_dataset = self.load_dataset(self.valid_data_paths)
    
        if self.inc == 1:
            self.train_mean = (
                self.train_dataset.tensors[0].mean() if self.train_mean is None else self.train_mean
            )
            self.train_std = (
                self.train_dataset.tensors[0].std() if self.train_std is None else self.train_std
            )

        elif self.inc == 2:
            self.train_mean = (
                self.train_dataset.tensors[0].mean(dim=(0, 2, 3))
                if self.train_mean is None
                else self.train_mean
            )
            self.train_std = (
                self.train_dataset.tensors[0].std(dim=(0, 2, 3))
                if self.train_std is None
                else self.train_std
            )
                
        if stage == 'fit' or stage is None:
            # Normalize training and validation datasets
            self.train_dataset = self.normalize_dataset(self.train_dataset)
            self.valid_dataset = self.normalize_dataset(self.valid_dataset)

        elif stage == 'test' or stage == 'predict':
            # Normalize test dataset
            self.test_dataset = self.normalize_dataset(self.test_dataset)

    def load_dataset(self, path):
        """
        Loads a dataset from a given path, either a string or an xarray Dataset object.

        Parameters
        ----------
        path : str or xr.Dataset
            Path to the dataset file or an xarray Dataset object.

        Returns
        -------
        dataset : TensorDataset
            A PyTorch TensorDataset containing the input data and labels.
        """
        if isinstance(path, str):
            ds = xr.open_dataset(path)
        elif isinstance(path, xr.Dataset):
            ds = path
        if self.inc == 1:
            # Single polarization (VV or VH)
            X = ds.sel(pol=self.pol).sigma0_detrend.values
            X = X[:, None, :, :]  # Add channel dimension
            y = ds.sel(pol=self.pol).ref_angles.values if 'ref_angles' in ds else np.zeros(len(X))

        elif self.inc == 2:
            # Dual polarization (VV, VH)
            X = ds.sigma0_detrend.values
            X = np.transpose(X, (0, 1, 2, 3))  # Transpose to (batch, channels, height, width)
            y = ds.ref_angles.values if 'ref_angles' in ds else np.zeros(X.shape[0])


        # Convert to PyTorch tensors
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        return TensorDataset(X_tensor, y_tensor)

    def normalize_dataset(self, dataset):
        """
        Normalizes the dataset using the mean and standard deviation of the training data.

        Args:
            dataset (TensorDataset): The dataset to normalize.

        Returns:
            TensorDataset: The normalized dataset.
        """
        X, y = dataset.tensors
        mean = self.train_mean
        std = self.train_std
        if self.inc == 2:
            mean = mean.view(1, -1, 1, 1)
            std = std.view(1, -1, 1, 1)
        X = (X - mean) / std
        return TensorDataset(X, y)

    def train_dataloader(self):
        """
        Creates and returns the DataLoader for the training dataset.

        Returns:
            DataLoader: DataLoader for the training dataset.
        """
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=4, shuffle=True)

    def val_dataloader(self):
        """
        Creates and returns the DataLoader for the validation dataset.

        Returns:
            DataLoader: DataLoader for the validation dataset.
        """
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, num_workers=4, shuffle=False)

    def test_dataloader(self):
        """
        Creates and returns the DataLoader for the test dataset.

        Returns:
            DataLoader: DataLoader for the test dataset.
        """
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=4, shuffle=False)
