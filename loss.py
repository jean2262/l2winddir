import torch
import torch.nn as nn


class CustomLoss(nn.Module):
    """
    Custom loss function for angular prediction tasks, designed to handle angular ambiguity.

    Args:
        ambiguite (bool, optional): If True, handles angular ambiguity by reducing the error range. Default is True.

    Attributes:
        ambi (bool): Indicates whether angular ambiguity handling is enabled.
    """

    def __init__(self, ambiguite=True):
        super(CustomLoss, self).__init__()
        self.ambi = ambiguite

    def forward(self, y_true, y_pred):
        """
        Computes the custom loss between the true and predicted values.

        Args:
            y_true (torch.Tensor): Ground truth angles (in radians).
            y_pred (torch.Tensor): Predicted angles (in radians).

        Returns:
            torch.Tensor: The computed loss value.
        """
        # Calculate the error between prediction and ground truth
        err = y_pred - y_true.view(-1, 1)

        # Handle angular ambiguity if specified
        if not self.ambi:
            err = err / 2

        # Compute the loss based on cosine similarity
        loss = 1 - torch.square(torch.cos(err))

        return loss.mean()
