import argparse
import os
import json
from src.utils.util import mkdir
import torch
import src


class ConfigParser:
    def __init__(self, set_master_gpu=True):
        parser = argparse.ArgumentParser()
        parser.add_argument('--exp_dir', type=str)
        args, _ = parser.parse_known_args()
        self._set_master_gpu = set_master_gpu

        self._exp_dir = args.exp_dir

        # parse default configuration
        self._parse_default()

        # overwrite default configuration with experiment specific config
        self._overwrite_default_opt()

        # prepare directories
        self._set_dirs()

        # set options
        self._init_opt()

    def get_config(self):
        return self._opt

    def _parse_default(self):
        # parse default config
        default_conf_path = os.path.join(os.path.dirname(src.__file__), "options", "config_default.json")
        with open(default_conf_path, 'r') as f:
            self._opt = json.load(f)

    def _overwrite_default_opt(self):
        # parse experiment specific config
        with open(os.path.join(self._exp_dir, 'config.json'), 'r') as f:
            specific_opt = json.load(f)

        # recursively overwrite options
        self._override_json(self._opt, specific_opt)

    def _override_json(self, default, specific):
        # recursively overwrite options
        for key, value in specific.items():
            if not isinstance(specific[key], dict):
                default[key] = specific[key]
            else:
                self._override_json(default[key], specific[key])

    def _set_dirs(self):
        # set necessary directories
        self._opt["dirs"] = {}
        self._opt["dirs"]["exp_dir"] = self._exp_dir
        self._opt["dirs"]["checkpoints"] = "checkpoints"
        self._opt["dirs"]["events"] = "events"
        self._opt["dirs"]["test"] = "test"

        # create necessary directories
        mkdir(os.path.join(self._opt["dirs"]["exp_dir"], self._opt["dirs"]["checkpoints"]))
        mkdir(os.path.join(self._opt["dirs"]["exp_dir"], self._opt["dirs"]["events"]))
        mkdir(os.path.join(self._opt["dirs"]["exp_dir"], self._opt["dirs"]["test"]))

    def _init_opt(self):
        # set load epoch conf
        self._set_and_check_load_epoch()

        # set selected gpus
        if self._set_master_gpu:
            self.set_gpus()

        # overwrite dataset parameters
        self._set_dataset_params()

        # print config
        self._print()

        return self._opt

    def _set_and_check_load_epoch(self):
        load_epoch = self._opt["model"]["load_epoch"]
        checkpoints_path = os.path.join(self._exp_dir, self._opt["dirs"]["checkpoints"])
        if os.path.exists(checkpoints_path):
            # if no epoch selected get the latest one (if any)
            if load_epoch == -1:
                load_epoch = 0
                for file in os.listdir(checkpoints_path):
                    if file.startswith("net_epoch_"):
                        load_epoch = max(load_epoch, int(file.split('_')[2]))
                self._opt["model"]["load_epoch"] = load_epoch

            # if epoch selected check that it exists
            else:
                found = False
                for file in os.listdir(checkpoints_path):
                    if file.startswith("net_epoch_"):
                        found = int(file.split('_')[2]) == load_epoch
                        if found: break
                assert found, 'Model for epoch %i not found' % load_epoch
        else:
            assert load_epoch < 1, 'Model for epoch %i not found' % load_epoch
            self._opt["model"]["load_epoch"] = 0

    def set_gpus(self):
        if torch.cuda.is_available():
            torch.cuda.set_device(self._opt["model"]["master_gpu"])

    def _set_dataset_params(self):
        # overwrite default dataset params with train, val, test specific ones
        for set in ("train", "val", "test"):
            dataset_name = "dataset_{}".format(set)
            tmp = self._opt["dataset"].copy()
            for x, v in self._opt[dataset_name].items():
                tmp[x] = v
            self._opt[dataset_name] = tmp

    def _print(self):
        print('------------ Options -------------')
        print(json.dumps(self._opt, indent=2))
        print('-------------- End ----------------')