import torch
import numpy as np
from src.transforms.transforms import TransformBase


class ToTensor(TransformBase):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, perkey_args, general_args):
        super(ToTensor, self).__init__()
        self._params = self._find_params_for_sample_key(perkey_args, general_args)

    def __call__(self, sample):
        for sample_key in sample.keys():
            if sample_key in self._params.keys():
                elem = sample[sample_key]
                if isinstance(elem, list):
                    for i, el in enumerate(elem):
                        elem[i] = self._to_tensor(el)
                else:
                    sample[sample_key] = self._to_tensor(elem)

        return sample

    def _to_tensor(self, sample):
        if sample.ndim == 2:
            sample = np.expand_dims(sample, -1)
        sample = sample.transpose((2, 0, 1))
        return torch.from_numpy(sample.astype(np.float32))

    def __str__(self):
        return 'ToTensor'


