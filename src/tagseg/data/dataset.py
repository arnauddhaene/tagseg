import logging

from torchvision import transforms

from kedro.extras.datasets.pickle import PickleDataSet
from kedro.extras.datasets.pandas import CSVDataSet


class TagSegDataSet(PickleDataSet):
    def __init__(self, *args, **kwargs):
        super(TagSegDataSet, self).__init__(*args, **kwargs)

    def _describe(self) -> str:
        return f"{self.__class__.__name__} in {'processed' if self._exists() else 'raw'} format."

    def _load(self, *args, **kwargs):
        log = logging.getLogger(__name__)

        if self._exists():
            log.info(f'Dataset exists and will be loaded from {self._filepath}')
            self._load_args = {}
            return super(TagSegDataSet, self)._load()
        else:
            data = self._load_except(**self._load_args)
            self._save(data)
            return data

    def _load_except(self):
        raise NotImplementedError

    def _preprocess_image(
        self, mu: float, sigma: float
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


class EvalInfoDataSet(CSVDataSet):
    def __init__(self, *args, **kwargs):
        super(EvalInfoDataSet, self).__init__(*args, **kwargs)

    def _describe(self) -> str:
        return f"{self.__class__.__name__} in {'processed' if self._exists() else 'raw'} format."

    def _load(self, *args, **kwargs):
        log = logging.getLogger(__name__)

        if self._exists():
            log.info(f'Evaluation Information exists and will be loaded from {self._filepath}')
            self._load_args = {}
            return super(EvalInfoDataSet, self)._load()
        else:
            data = self._load_except(**self._load_args)
            self._save(data)
            return data

    def _load_except(self):
        raise NotImplementedError
