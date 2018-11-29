from __future__ import division

import numpy as np
from src.transforms.transforms import TransformBase
from skimage.transform import resize as sk_resize
import cv2

class FixedResize(TransformBase):
    """Resize the image and the ground truth to specified resolution.
    Args:
    """

    def __init__(self, perkey_args, general_args):
        super(FixedResize, self).__init__()
        self._params = self._find_params_for_sample_key(perkey_args, general_args)

    def __call__(self, sample):
        for sample_key in sample.keys():
            if sample_key in self._params.keys():
                args = self._params[sample_key]
                interpolation = self._interpolation_str_to_cv2(args["interpolation"])
                if isinstance(sample[sample_key], list):
                    sample[sample_key] = list(map(lambda x: self._fixed_resize(x, tuple(args["resolution"]), flagval=interpolation, scikit=args["scikit"]), sample[sample_key]))

                else:
                    sample[sample_key] = self._fixed_resize(sample[sample_key], tuple(args["resolution"]),
                                                      flagval=interpolation, scikit=args["scikit"])

        return sample

    def _fixed_resize(self, sample, resolution, flagval, scikit=False):
        if isinstance(resolution, int):
            tmp = [resolution, resolution]
            tmp[np.argmax(sample.shape[:2])] = int(round(float(resolution)/np.min(sample.shape[:2])*np.max(sample.shape[:2])))
            resolution = tuple(tmp)

        if sample.ndim == 2 or (sample.ndim == 3 and sample.shape[2] == 3):
            if scikit:
                sample = sk_resize(sample, resolution,  order=0, mode='constant').astype(sample.dtype)
            else:
                sample = cv2.resize(sample, resolution[::-1], interpolation=flagval)
        else:
            tmp = sample
            sample = np.zeros(np.append(resolution, tmp.shape[2]), dtype=np.float32)
            for ii in range(sample.shape[2]):
                sample[:, :, ii] = cv2.resize(tmp[:, :, ii], resolution[::-1], interpolation=flagval)
        return sample

    def __str__(self):
        return 'FixedResize:' + str(self._params)
