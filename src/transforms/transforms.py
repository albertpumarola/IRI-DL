import cv2


class TransformsFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_by_name(transform_name, perkey_args, general_args):
        if transform_name == 'fixed_resize':
            from .fixed_resize import FixedResize
            network = FixedResize(perkey_args, general_args)
        elif transform_name == 'to_tensor':
            from .to_tensor import ToTensor
            network = ToTensor(perkey_args, general_args)
        elif transform_name == 'normalize':
            from .normalize import Normalize
            network = Normalize(perkey_args, general_args)
        elif transform_name == "rand_horz_flip":
            from .rand_horz_fip import RandomHorizontalFlip
            network = RandomHorizontalFlip(perkey_args, general_args)
        elif transform_name == "rand_scale_n_rotate":
            from .rand_scale_n_rotate import RandScaleNRotate
            network = RandScaleNRotate(perkey_args, general_args)
        else:
            raise ValueError("Transform %s not recognized." % transform_name)

        return network


class TransformBase(object):
    def __init__(self):
        super(TransformBase, self).__init__()
        self._name = 'TransformBase'

    @property
    def name(self):
        return self._name

    def _interpolation_str_to_cv2(self, interp_str):
        if interp_str == 'nearest':
            interp = cv2.INTER_NEAREST
        elif interp_str == "linear":
            interp = cv2.INTER_LINEAR
        elif interp_str == "cubic":
            interp = cv2.INTER_CUBIC
        else:
            raise ValueError("Interpolation not available")
        return interp

    def _find_params_for_sample_key(self, sample_keys, args):
        params = {}
        keys = sample_keys.keys()
        for key in keys:
            key_params = args.copy()
            for arg_key, value in sample_keys[key].copy().items():
                key_params[arg_key] = value
            params[key] = key_params
        return params