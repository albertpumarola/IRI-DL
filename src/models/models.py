import os
import torch
from collections import OrderedDict
import time


class ModelsFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_by_name(model_name, *args, **kwargs):
        if model_name == 'model1':
            from .model1 import Model1
            model = Model1(*args, **kwargs)
        else:
            raise ValueError("Model %s not recognized." % model_name)

        return model


class BaseModel(object):

    def __init__(self, opt):
        self._name = 'BaseModel'

        self._opt = opt

        self._master_gpu_id = opt["model"]["master_gpu"]
        self._reg_gpus_ids = opt["model"]["reg_gpus"]

        self._device_master = f"cuda:{self._master_gpu_id}" if torch.cuda.is_available() else "cpu"
        self._reg_device_master = f"cuda:{self._reg_gpus_ids[0]}" if torch.cuda.is_available() else "cpu"

        self._is_train = opt["model"]["is_train"]
        self._dataset_type = "dataset_train" if self._is_train else "dataset_test"
        self._saved_files = {"checkpoint": dict()}

        self._save_dir = os.path.join(opt["dirs"]["exp_dir"], opt["dirs"]["checkpoints"])


    @property
    def name(self):
        return self._name

    @property
    def is_train(self):
        return self._is_train

    def set_input(self, input):
        raise NotImplementedError

    def set_train(self):
        raise NotImplementedError

    def set_eval(self):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError

    def forward(self, keep_data_for_visuals=False):
        raise NotImplementedError

    # used in test time, no backprop
    def test(self):
        raise NotImplementedError

    def get_image_paths(self):
        return {}

    def optimize_parameters(self):
        raise NotImplementedError

    def get_current_visuals(self):
        return {}

    def get_current_errors(self):
        return {}

    def get_current_scalars(self):
        return {}

    def get_current_histograms(self):
        return {}

    def save(self, label, save_type, do_remove_prev):
        raise NotImplementedError

    def remove(self, epoch_label):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError

    def _save_optimizer(self, optimizer, optimizer_label, epoch_label, save_type, do_remove_prev):
        save_filename = 'opt_epoch_%s_id_%s.pth' % (epoch_label, optimizer_label)
        save_path = os.path.join(self._save_dir, save_filename)
        torch.save(optimizer.state_dict(), save_path)
        print('saved optimizer: %s' % save_path)
        self._update_and_remove_previous_file(optimizer_label, save_path, save_type, do_remove_prev)

    def _load_optimizer(self, optimizer, optimizer_label, epoch_label):
        load_filename = 'opt_epoch_%s_id_%s.pth' % (epoch_label, optimizer_label)
        load_path = os.path.join(self._save_dir, load_filename)
        if os.path.exists(load_path):
            optimizer.load_state_dict(torch.load(load_path))
            print('loaded optimizer: %s' % load_path)
        else:
            print('NOT!! loaded optimizer: %s' % load_path)
        self._update_and_remove_previous_file(optimizer_label, load_path, "checkpoint", False)

    def _save_network(self, network, network_label, epoch_label, save_type, do_remove_prev):
        save_filename = 'net_epoch_%s_id_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self._save_dir, save_filename)
        torch.save(network.state_dict(), save_path)
        print('saved net: %s' % save_path)
        self._update_and_remove_previous_file(network_label, save_path, save_type, do_remove_prev)

    def _load_network(self, network, network_label, epoch_label):
        load_filename = 'net_epoch_%s_id_%s.pth' % (epoch_label, network_label)
        load_path = os.path.join(self._save_dir, load_filename)

        if os.path.exists(load_path):
            loaded_model = torch.load(load_path, map_location=lambda storage, loc: storage)
            if isinstance(network, torch.nn.parallel.DataParallel) and 'module.' in list(loaded_model.keys())[0]:
                network.load_state_dict(loaded_model)
            elif isinstance(network, torch.nn.parallel.DataParallel) and 'module.' not in list(loaded_model.keys())[0]:
                new_state_dict = OrderedDict()
                for k, v in loaded_model.items():
                    name = 'module.' + k  # add `module.`
                    new_state_dict[name] = v
                network.load_state_dict(new_state_dict)
            else:
                if 'module.' in list(loaded_model.keys())[0]:
                    new_state_dict = OrderedDict()
                    for k, v in loaded_model.items():
                        name = k[7:]  # remove `module.`
                        new_state_dict[name] = v
                else:
                    new_state_dict = loaded_model

                network.load_state_dict(new_state_dict)
            print('loaded net: %s' % load_path)
        else:
            print('NOT!! loaded net: %s' % load_path)
        self._update_and_remove_previous_file(network_label, load_path, "checkpoint", False)

    def _update_and_remove_previous_file(self, label, new_file_path, save_type, do_remove_prev):
        if save_type not in self._saved_files:
            raise NotImplementedError

        ref_dict = self._saved_files[save_type]
        if label in ref_dict:
            previous_file_path = ref_dict[label]
            if do_remove_prev:
                if os.path.exists(previous_file_path) and new_file_path != previous_file_path:
                    os.remove(previous_file_path)
                    print('delete file: %s' % previous_file_path)
        ref_dict[label] = new_file_path

    def update_learning_rate(self, curr_epoch):
        pass

    def _lr_linear(self, current_lr, nepochs_decay, base_lr):
        return current_lr - base_lr / nepochs_decay

    def _update_learning_rate(self, optimizer, label, curr_lr, new_lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
        print('update %s learning rate: %f -> %f' % (label, curr_lr, new_lr))