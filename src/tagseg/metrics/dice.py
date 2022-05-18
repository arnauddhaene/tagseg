import torch


class DiceMetric():

    def __init__(self, include_background: bool = True):
        self.offset = 1 if include_background else 0

    @staticmethod
    def dice_coefficient(y_pred: torch.Tensor, y: torch.Tensor):

        if not isinstance(y_pred, torch.Tensor):
            raise TypeError(f"y_pred type is not a torch.Tensor. Got {type(input)}")

        if not len(y_pred.shape) == 4:
            raise ValueError(
                f"Invalid y_pred shape, we expect BxNxHxW. \
                Got: {y_pred.shape}"
            )

        if not y_pred.shape == y.shape:
            raise ValueError(
                f"y_pred and y shapes must be the same. \
                Got: {y_pred.shape} and {y.shape}"
            )

        if not y_pred.device == y.device:
            raise ValueError(
                f"y_pred and y must be on the same device. \
                Got: {y_pred.device} and {y.device}"
            )

        dims = (2, 3)
        intersection = torch.sum(y_pred * y, dims)
        cardinality = torch.sum(y_pred + y, dims)

        return torch.mean(2.0 * intersection / (cardinality + 1e-8), dim=0)

    def __call__(self, y_pred: torch.Tensor, y: torch.Tensor):
        return self.dice_coefficient(y_pred, y)[self.offset:].mean()
