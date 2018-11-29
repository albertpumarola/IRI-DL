from __future__ import division

import numpy.random as random
import cv2
from src.transforms.transforms import TransformBase

class RandomHorizontalFlip(TransformBase):
    """Horizontally flip the given image and ground truth randomly with a probability of 0.5."""

    def __init__(self, perkey_args, general_args):
        super(RandomHorizontalFlip, self).__init__()
        self._params = self._find_params_for_sample_key(perkey_args, general_args)

    def __call__(self, sample):
        do_flip = random.random() < 0.5
        for sample_key in sample.keys():
            if sample_key in self._params.keys():
                elem = sample[sample_key]

                if isinstance(elem, list):
                    for i, el in enumerate(elem):
                        elem[i] = self._flip(el, do_flip)
                else:
                    sample[sample_key] = self._flip(elem, do_flip)

        return sample

    def _flip(self, elem, do_flip):
        if do_flip:
            elem = cv2.flip(elem, flipCode=1)
        return elem

    def __str__(self):
        return 'RandomHorizontalFlip'