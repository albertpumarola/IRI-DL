import torch.utils.data
from src.data.dataset import DatasetFactory
from src.transforms.transforms import TransformsFactory
import torchvision.transforms as transforms


class CustomDatasetDataLoader:
    def __init__(self, opt, is_for="train", subset=None):
        self._opt = opt
        self._is_for = is_for
        self._subset = is_for if subset is None else subset

        # create transforms
        transform = self._create_transform()

        # create dataset and dataloader
        self._create_dataloader(transform)

    def _create_transform(self):
        # get desired transforms
        transform_list = self._opt["transforms_{}".format(self._is_for)]

        # create desired transforms
        tfs = []
        for tf_key in transform_list:
            # get transform params
            tf_info = self._opt["transforms"][tf_key]
            perkey_args = tf_info["perkey_args"] if "perkey_args" in tf_info else None
            general_args = tf_info["general_args"] if "general_args" in tf_info else None

            # create transform
            tf = TransformsFactory.get_by_name(tf_info["type"], perkey_args, general_args)
            tfs.append(tf)
        return transforms.Compose(tfs)

    def _create_dataloader(self, transform):
        # create dataset
        dataset_type = "dataset_{}".format(self._is_for)
        self._batch_size = self._opt[dataset_type]["batch_size"]
        self._dataset = DatasetFactory.get_by_name(self._opt[dataset_type]["type"], self._opt, self._is_for, self._subset, transform, dataset_type)

        # create dataloader
        self._dataloader = torch.utils.data.DataLoader(
            self._dataset,
            batch_size=self._batch_size,
            shuffle=not self._opt[dataset_type]["serial_batches"],
            num_workers=self._opt[dataset_type]["n_threads"],
            drop_last=bool(self._opt[dataset_type]["drop_last_batch"]),
            pin_memory=True)

    def get_batch_size(self):
        return self._batch_size

    def get_dataset(self):
        return self._dataset

    def load_data(self):
        return self._dataloader

    def __len__(self):
        return len(self._dataset)
