from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from loss import CustomLoss
from einops.layers.torch import Reduce
import math
import numpy as np

# Set higher precision for matrix multiplication
torch.set_float32_matmul_precision('high')


def simple_conv(inc, outc, hid, depth, down, act=nn.ReLU(), drop=False):
    """
    Creates a simple convolutional neural network.

    Args:
        inc (int): Number of input channels.
        outc (int): Number of output channels.
        hid (int): Number of hidden units.
        depth (int): Number of convolutional layers.
        down (int): Downscaling factor.
        act (nn.Module): Activation function.
        drop (bool): Whether to include dropout layers.

    Returns:
        nn.Sequential: The convolutional neural network model.
    """
    layers = [
        nn.InstanceNorm2d(inc),
        nn.Conv2d(inc, hid, kernel_size=3, padding=1),
    ]
    if down > 1:
        layers.append(Reduce('batch channel (h h2) (w w2) -> batch channel h w', reduction='max', w2=down, h2=down))

    for _ in range(depth):
        layers.append(nn.Sequential(
            nn.Conv2d(hid, hid, kernel_size=3, padding=1),
            nn.BatchNorm2d(hid),
            nn.Dropout(p=0.3) if drop else nn.Identity(),
            act,
        ))

    layers.append(Reduce('batch channel h w -> batch channel', reduction='mean'))
    layers.append(nn.Linear(hid, outc))

    return nn.Sequential(*layers)


# Constants for Gaussian distributions
ONEOVERSQRT2PI = 1.0 / math.sqrt(2 * math.pi)
LOG2PI = math.log(2 * math.pi)


class MDN(nn.Module):
    """
    Mixture Density Network (MDN) to predict parameters of a mixture of Gaussians.

    Args:
        mod (nn.Module): The base neural network model.
        num_gaussians (int): Number of Gaussian components.

    Attributes:
        mod (nn.Module): The base neural network model.
        num_gaussians (int): Number of Gaussian components.
        pi (nn.Linear): Linear layer to predict mixture coefficients.
        sigma (nn.Linear): Linear layer to predict standard deviations.
        mu (nn.Linear): Linear layer to predict means.
    """

    def __init__(self, mod, num_gaussians):
        super(MDN, self).__init__()
        self.mod = mod
        self.num_gaussians = num_gaussians
        self.pi = nn.Linear(self.mod[-1].out_features, num_gaussians)
        self.sigma = nn.Linear(self.mod[-1].out_features, num_gaussians)
        self.mu = nn.Linear(self.mod[-1].out_features, num_gaussians)

    def forward(self, x):
        """
        Forward pass through the MDN.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            tuple: The mixture coefficients (pi), standard deviations (sigma), and means (mu).
        """
        features = self.mod(x)
        pi = torch.softmax(self.pi(features), dim=-1)
        sigma = torch.exp(self.sigma(features)) + 1e-6
        mu = self.mu(features)
        return pi, sigma, mu

    def gaussian_probability(self, sigma, mu, y):
        """
        Computes the Gaussian log probability for each component.

        Args:
            sigma (torch.Tensor): Standard deviations.
            mu (torch.Tensor): Means.
            y (torch.Tensor): Target values.

        Returns:
            torch.Tensor: Log probabilities for each component.
        """
        target = y.view(-1, 1).expand_as(sigma)
        p0 = -torch.log(sigma) - 0.5 * LOG2PI - 0.5 * torch.pow((target + math.pi - mu) / sigma, 2)
        p1 = -torch.log(sigma) - 0.5 * LOG2PI - 0.5 * torch.pow((target - mu) / sigma, 2)
        p2 = -torch.log(sigma) - 0.5 * LOG2PI - 0.5 * torch.pow((target - math.pi - mu) / sigma, 2)

        combined_log_prob = torch.logsumexp(torch.stack([p0, p1, p2], dim=-1), dim=-1)
        return combined_log_prob

    def log_prob(self, pi, sigma, mu, y):
        """
        Computes the log probability of the mixture model.

        Args:
            pi (torch.Tensor): Mixture coefficients.
            sigma (torch.Tensor): Standard deviations.
            mu (torch.Tensor): Means.
            y (torch.Tensor): Target values.

        Returns:
            torch.Tensor: Log probability of the mixture model.
        """
        log_component_prob = self.gaussian_probability(sigma, mu, y)
        log_mix_prob = torch.log(F.gumbel_softmax(pi, tau=1, dim=-1) + 1e-15)
        return torch.logsumexp(log_component_prob + log_mix_prob, dim=-1)


class MyModel(L.LightningModule):
    """
    Custom model that integrates a convolutional backbone with a Mixture Density Network (MDN).

    Args:
        inc (int): Number of input channels.
        hid (int): Number of hidden units.
        depth (int): Number of convolutional layers.
        down (int): Downscaling factor.
        model_mdn (bool): Whether to use a Mixture Density Network.
        num_gaussians (int): Number of Gaussian components for the MDN.
        drop (bool): Whether to include dropout layers.
        criterion (nn.Module): Loss function.
        opt_fn (function): Optimizer function.

    Attributes:
        mod (nn.Sequential): Convolutional model.
        mdn (MDN): Mixture Density Network.
        train_mean (float): Mean of training data.
        train_std (float): Standard deviation of training data.
        criterion (nn.Module): Loss function.
        opt_fn (function): Optimizer function.
    """

    def __init__(
            self,
            inc=2,
            hid=128,
            depth=7,
            down=4,
            model_mdn=True,
            num_gaussians=2,
            drop=False,
            criterion=CustomLoss(),
            weight_decay=None,
            lr=1e-3,
            # data_augmentation=False,
    ):
        super().__init__()
        self.criterion = criterion
        self.weight_decay = weight_decay
        self.lr = lr
        self.train_mean = None
        self.train_std = None
        self.model_mdn = model_mdn
        # self.data_augmentation = data_augmentation
        self.outc = 3 if model_mdn else 1
        self.mod = simple_conv(inc=inc, outc=self.outc, hid=hid, depth=depth, down=down, act=nn.ReLU(), drop=drop)
        if model_mdn:
            self.mdn = MDN(mod=self.mod, num_gaussians=num_gaussians)

    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            tuple: Predicted values, uncertainty, and MDN outputs (pi, sigma, mu) if model_mdn=True.
        """
            
        if self.model_mdn:
            pi, sigma, mu = self.mdn(x)
            pred = mu.gather(-1, pi.argmax(dim=-1, keepdim=True))
            uncertainty = sigma.gather(-1, pi.argmax(dim=-1, keepdim=True))
            return pred, uncertainty
        else:
            return self.mod(x)

    def set_normalization_params(self, mean, std):
        """
        Sets the normalization parameters for the input data.

        Args:
            mean (float): Mean of training data.
            std (float): Standard deviation of training data.
        """
        self.train_mean = mean
        self.train_std = std

    def mdn_loss(self, pi, sigma, mu, y):
        """
        Computes the loss for the MDN.

        Args:
            pi (torch.Tensor): Mixture coefficients.
            sigma (torch.Tensor): Standard deviations.
            mu (torch.Tensor): Means.
            y (torch.Tensor): Target values.

        Returns:
            torch.Tensor: The MDN loss.
        """
        log_prob = self.mdn.log_prob(pi, sigma, mu, y)
        return -log_prob.mean()

    def training_step(self, batch):
        """
        Training step logic.

        Args:
            batch (tuple): A batch of data (inputs, targets).

        Returns:
            torch.Tensor: The computed loss.
        """
        x, y = batch
        # if self.data_augmentation:
        #     x = torch.rot90(x, np.random.choice([0, 2]), [2,3])
        if self.model_mdn:
            pred, uncertainty = self(x)
            pi, sigma, mu = self.mdn(x)
            mdn_loss = self.mdn_loss(pi, sigma, mu, y)
            loss = self.criterion(y, pred)
            combined_loss = mdn_loss + loss
            self.log('train_loss', loss, prog_bar=True)
            self.log('train_loss_mdn', mdn_loss, prog_bar=True)
            return combined_loss
        else:
            pred = self(x)
            loss = self.criterion(y, pred)
            self.log('train_loss', loss)
            return loss

    def err_calculation(self, batch):
        """
        Error calculation logic for validation and test steps.

        Args:
            batch (tuple): A batch of data (inputs, targets).

        Returns:
            tuple: MDN loss, criterion loss, error, and MDN outputs (pi, sigma, mu) if model_mdn=True.
        """
        x, y = batch
        if self.model_mdn:
            pred, uncertainty = self(x)
            pi, sigma, mu = self.mdn(x)
            mdn_loss = self.mdn_loss(pi, sigma, mu, y)
        else:
            pred = self(x)

        loss = self.criterion(y, pred)

        # Angular error calculation
        ang = 180
        y_pred_deg = torch.rad2deg(pred) % ang
        y_deg = torch.rad2deg(y) % ang
        angular_diff = torch.abs(y_deg.view(-1, 1) - y_pred_deg)
        err = torch.min(angular_diff, ang - angular_diff).mean()
        if self.model_mdn:
            return mdn_loss, loss, err
        else:
            return loss, err

    def validation_step(self, batch):
        """
        Validation step logic.

        Args:
            batch (tuple): A batch of data (inputs, targets).

        Returns:
            torch.Tensor: The computed loss.
        """
        if self.model_mdn:
            mdn_loss, loss, err = self.err_calculation(batch)
            combined_loss = mdn_loss + loss
            self.log('val_loss_mdn', mdn_loss, prog_bar=True)
        else:
            loss, err = self.err_calculation(batch)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_err_180', err, prog_bar=True)
        if self.model_mdn:
            return combined_loss
        else:
            return loss

    def test_step(self, batch):
        """
        Test step logic.

        Args:
            batch (tuple): A batch of data (inputs, targets).

        Returns:
            torch.Tensor: The computed loss.
        """
        if self.model_mdn:
            mdn_loss, loss, err = self.err_calculation(batch)
            combined_loss = mdn_loss + loss
            self.log('test_loss_mdn', mdn_loss, prog_bar=True)
        else:
            loss, err = self.err_calculation(batch)
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_err_180', err, prog_bar=True)
        if self.model_mdn:
            return combined_loss
        else:
            return loss

    def predict_step(self, batch):
        """
        Prediction step logic.

        Args:
            batch (tuple): A batch of data (inputs, targets).

        Returns:
            torch.Tensor: The predicted values and uncertainties.
        """
        x, y = batch
        return self(x).cpu()

        
    def configure_optimizers(self):
        """
        Configures the optimizer and learning rate scheduler.

        Returns:
            dict: A dictionary containing the optimizer, learning rate scheduler,
            and the metric to monitor ('val_loss').
        """
        weight_decay = self.weight_decay
        if isinstance(weight_decay, str) and weight_decay.lower() == "none":
            weight_decay = None
        else:
            weight_decay = float(weight_decay)
            
        if weight_decay:
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

