from __future__ import division

import numpy.random as random
import cv2
from vois_gan.transforms.transforms import TransformBase


class RandScaleNRotate(TransformBase):
    """Scale (zoom-in, zoom-out) and Rotate the image and the ground truth.
    Args:
        two possibilities:
        1.  rots (tuple): (minimum, maximum) rotation angle
            scales (tuple): (minimum, maximum) scale
        2.  rots [list]: list of fixed possible rotation angles
            scales [list]: list of fixed possible scales
    """
    def __init__(self, perkey_args, general_args):
        super(RandScaleNRotate, self).__init__()
        self._params = self._find_params_for_sample_key(perkey_args, general_args)

    def __call__(self, sample):
        for sample_key in self._params.keys():
            data_ids = self._params[sample_key]["data_ids"]
            data_interplts = self._params[sample_key]["data_interpolations"]
            assert len(data_ids) == len(data_interplts)

            data_0 = sample[data_ids[0]]

            if isinstance(data_0, list):
                for i in range(len(data_0)):
                    rand_rot = self._rand_rot(self._params[sample_key]["rots"])
                    rand_scales = self._rand_scale(self._params[sample_key]["scales"])
                    for data_id, data_interplt in zip(data_ids, data_interplts):
                        if data_id in sample.keys():
                            interpolation = self._interpolation_str_to_cv2(data_interplt)
                            sample[data_id][i] = self._transform(sample[data_id][i], rand_rot, rand_scales, interpolation)

            else:
                rand_rot = self._rand_rot(self._params[sample_key]["rots"])
                rand_scales = self._rand_scale(self._params[sample_key]["scales"])

                for data_id, data_interplt in zip(data_ids, data_interplts):
                    if data_id in sample.keys():
                        interpolation = self._interpolation_str_to_cv2(data_interplt)
                        sample[data_id] = self._transform(sample[data_id], rand_rot, rand_scales, interpolation)

        return sample

    def _rand_rot(self, rots):
        return (rots[1] - rots[0]) * random.random() - (rots[1] - rots[0]) / 2

    def _rand_scale(self, scales):
        return (scales[1] - scales[0]) * random.random() - (scales[1] - scales[0]) / 2 + 1

    def _transform(selfs, sample, rot, sc, flagval):
        h, w = sample.shape[:2]
        center = (w / 2, h / 2)
        assert (center != 0)  # Strange behaviour warpAffine
        M = cv2.getRotationMatrix2D(center, rot, sc)
        return cv2.warpAffine(sample, M, (w, h), flags=flagval)

    def __str__(self):
        return 'ScaleNRotate:' + str(self._params)