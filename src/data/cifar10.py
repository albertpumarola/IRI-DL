import os.path
from src.data.dataset import DatasetBase
import numpy as np
from torchvision.datasets.utils import download_url
import tarfile
import sys
import pickle

class Cifar10Dataset(DatasetBase):
    def __init__(self, opt, is_for, subset, transform, dataset_type):
        super(Cifar10Dataset, self).__init__(opt, is_for, subset, transform, dataset_type)
        self._name = 'cifar10'

        # init meta
        self._init_meta(opt)

        # download dataset if necessary
        self._download()

        # read dataset
        self._read_dataset()

        # read meta
        self._read_meta()

    def _init_meta(self, opt):
        self._rgb = not opt[self._name]["use_bgr"]
        self._root = opt[self._name]["data_dir"]
        self._data_folder = opt[self._name]["data_folder"]
        self._meta_file = opt[self._name]["meta_file"]
        self._url = opt[self._name]["url"]
        self._filename = opt[self._name]["filename"]
        self._tgz_md5 = opt[self._name]["tgz_md5"]

        if self._is_for == "train":
            self._ids_filename = self._opt[self._name]["train_ids_file"]
        elif self._is_for == "val":
            self._ids_filename = self._opt[self._name]["val_ids_file"]
        elif self._is_for == "test":
            self._ids_filename = self._opt[self._name]["test_ids_file"]
        else:
            raise ValueError(f"is_for={self._is_for} not valid")

    def __getitem__(self, index):
        assert (index < self._dataset_size)

        # get data
        img, target = self._data[index], self._targets[index]

        # pack data
        sample = {'img': img, 'target': target}

        # apply transformations
        if self._transform is not None:
            sample = self._transform(sample)

        return sample

    def __len__(self):
        return self._dataset_size

    def _read_dataset(self):
        # read ids
        use_ids_filepath = os.path.join(self._root, self._ids_filename)
        valid_ids_root = self._read_valid_ids(use_ids_filepath)

        # load the picked numpy arrays
        self._data = []
        self._targets = []
        for file_name in valid_ids_root:
            file_path = os.path.join(self._root, self._data_folder, file_name)
            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')
                self._data.append(entry['data'])
                if 'labels' in entry:
                    self._targets.extend(entry['labels'])
                else:
                    self._targets.extend(entry['fine_labels'])

        # reshape data
        self._data = np.vstack(self._data).reshape(-1, 3, 32, 32)
        self._data = self._data.transpose((0, 2, 3, 1))  # convert to HWC

        # dataset size
        self._dataset_size = len(self._data)

    def _read_meta(self):
        path = os.path.join(self._root, self._data_folder, self._meta_file)
        with open(path, 'rb') as infile:
            if sys.version_info[0] == 2:
                data = pickle.load(infile)
            else:
                data = pickle.load(infile, encoding='latin1')
            self.classes = data["label_names"]
        self._class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def _read_valid_ids(self, file_path):
        ids = np.loadtxt(file_path, dtype=np.str)
        return np.expand_dims(ids, 0) if ids.ndim == 0 else ids

    def _download(self):
        # check already downloaded
        if os.path.isdir(os.path.join(self._root, self._data_folder)):
            return

        # download file
        print("It will take aprox 15 min...")
        download_url(self._url, self._root, self._filename, self._tgz_md5)

        # extract file
        with tarfile.open(os.path.join(self._root, self._filename), "r:gz") as tar:
            tar.extractall(path=self._root)