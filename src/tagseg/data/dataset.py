
from kedro.extras.datasets.pickle import PickleDataSet

import torch
from torchvision import transforms


class TagSegDataSet(PickleDataSet):

    def __init__(self, *args, **kwargs):
        super(TagSegDataSet, self).__init__(*args, **kwargs)

    def _describe(self) -> str:
        return ""

    def _load(self, *args, **kwargs):
        if self._exists():
            self._load_args = {}
            return super(TagSegDataSet, self)._load()
        else:
            data = self._load_except(**self._load_args)
            self._save(data)
            return data

    def _load_except(self):
        raise NotImplementedError

    def _preprocess_image(
        self, mu: float, sigma: float, label: torch.Tensor = None
    ) -> transforms.Compose:
        """Preprocess image

        Args:
            mu (float): average for normalization layer
            sigma (float): standard deviation for normalization layer

        Returns:
            transforms.Compose: transformation callback function
        """
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=mu, std=sigma),
                transforms.Resize((256, 256)),
            ]
        )

    def _preprocess_label(self) -> transforms.Compose:
        """Preprocess mask

        Returns:
            transforms.Compose: transformation callback function
        """
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(
                    (256, 256), interpolation=transforms.InterpolationMode.NEAREST
                ),
            ]
        )
