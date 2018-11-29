import torch.nn as nn

class LossFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_by_name(loss_name, *args, **kwargs):
        raise ValueError(f"Loss %s not recognized." % loss_name)


class LossBase(nn.Module):
    def __init__(self):
        super(LossBase, self).__init__()
        self._name = 'BaseLoss'

    @property
    def name(self):
        return self._name