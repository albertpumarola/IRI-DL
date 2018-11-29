from src.options.config_parser import ConfigParser
from src.data.custom_dataset_data_loader import CustomDatasetDataLoader
from src.models.models import ModelsFactory
from src.utils.tb_visualizer import TBVisualizer
import torch
import torch.backends.cudnn as cudnn
from src.utils.util import append_dictionaries, mean_dictionary
import numpy as np
import time

class Train:
    def __init__(self):
        self._opt = ConfigParser().get_config()
        self._opt["model"]["is_train"] = True
        self._get_conf_params()
        cudnn.benchmark = True

        # create visualizer
        self._tb_visualizer = TBVisualizer(self._opt)

        # prepare data
        self._prepare_data()

        # check options
        self._check_options()

        # create model
        model_type = self._opt["model"]["type"]
        self._model = ModelsFactory.get_by_name(model_type, self._opt)

        # start train
        self._train()

    def _prepare_data(self):
        # create dataloaders
        data_loader_train = CustomDatasetDataLoader(self._opt, is_for="train")
        data_loader_val = CustomDatasetDataLoader(self._opt, is_for="val")

        # create dataset
        self._dataset_train = data_loader_train.load_data()
        self._dataset_val = data_loader_val.load_data()

        # get dataset properties
        self._dataset_train_size = len(data_loader_train)
        self._dataset_val_size = len(data_loader_val)
        self._num_batches_per_epoch = len(self._dataset_train)

        # create visualizer
        self._tb_visualizer.print_msg('#train images = %d' % self._dataset_train_size)
        self._tb_visualizer.print_msg('#val images = %d' % self._dataset_val_size)

        # get batch size
        self._train_batch_size = data_loader_train.get_batch_size()
        self._val_batch_size = data_loader_val.get_batch_size()

    def _get_conf_params(self):
        self._load_epoch = self._opt["model"]["load_epoch"]
        self._nepochs_no_decay = self._opt["train"]["nepochs_no_decay"]
        self._nepochs_decay = self._opt["train"]["nepochs_decay"]
        self._print_freq_s = self._opt["logs"]["print_freq_s"]
        self._save_latest_freq_s = self._opt["logs"]["save_latest_freq_s"]
        self._display_freq_s = self._opt["logs"]["display_freq_s"]
        self._num_iters_validate = self._opt["train"]["num_iters_validate"]

    def _check_options(self):
        assert self._opt["dataset_train"]["batch_size"] == self._opt["dataset_val"]["batch_size"], \
            "batch for val and train are required to be equal"

    def _train(self):
        # meta
        self._total_steps = self._load_epoch * self._dataset_train_size
        self._iters_per_epoch = len(self._dataset_train)
        self._last_display_time = None
        self._last_save_latest_time = None
        self._last_print_time = time.time()

        self._i_epoch = self._load_epoch + 1
        self._model.update_learning_rate(max(self._load_epoch + 1, 1))

        for i_epoch in range(self._load_epoch + 1, self._nepochs_no_decay + self._nepochs_decay + 1):
            self._i_epoch = i_epoch
            epoch_start_time = time.time()

            # train epoch
            self._train_epoch(i_epoch)
            self._model.save(i_epoch, "checkpoint")

            # print epoch info
            time_epoch = time.time() - epoch_start_time
            self._tb_visualizer.print_msg('End of epoch %d / %d \t Time Taken: %d sec (%d min or %d h)' %
                  (i_epoch, self._nepochs_no_decay + self._nepochs_decay, time_epoch,
                   time_epoch / 60, time_epoch / 3600))

            # print epoch error
            self._display_visualizer_avg_epoch(i_epoch)

            # update learning rate
            self._model.update_learning_rate(i_epoch+1)

    def _train_epoch(self, i_epoch):
        self._model.set_train()

        # meta
        iter_start_time = time.time()
        iter_read_time = 0
        iter_procs_time = 0
        num_iters_time = 0
        self._epoch_train_e = dict()
        self._epoch_val_e = dict()

        for i_train_batch, train_batch in enumerate(self._dataset_train):
            iter_read_time += (time.time() - iter_start_time) / self._train_batch_size
            iter_after_read_time = time.time()

            # display flags
            do_visuals = self._last_display_time is None or\
                         time.time() - self._last_display_time > self._display_freq_s or\
                         i_train_batch == self._num_batches_per_epoch-1
            do_print_terminal = time.time() - self._last_print_time > self._print_freq_s or do_visuals

            # train model
            self._model.set_input(train_batch)
            self._model.optimize_parameters(keep_data_for_visuals=do_visuals)

            # update epoch info
            iter_procs_time += (time.time() - iter_after_read_time) / self._train_batch_size
            self._total_steps += self._train_batch_size
            num_iters_time += 1

            # display terminal
            if do_print_terminal:
                iter_read_time /= num_iters_time
                iter_procs_time /= num_iters_time
                self._display_terminal(iter_read_time, iter_procs_time, i_epoch, i_train_batch, do_visuals)
                self._last_print_time = time.time()

            # display visualizer
            if do_visuals:
                self._display_visualizer_train(self._total_steps, iter_read_time, iter_procs_time)
                self._display_visualizer_val(i_epoch, self._total_steps)
                self._last_display_time = time.time()

            # save model
            if self._last_save_latest_time is None or time.time() - self._last_save_latest_time > self._save_latest_freq_s:
                self._model.save(i_epoch, "checkpoint")
                self._last_save_latest_time = time.time()

            # reset metadata time
            if do_print_terminal:
                iter_read_time = 0
                iter_procs_time = 0
                num_iters_time = 0

            iter_start_time = time.time()

    def _display_terminal(self, iter_read_time, iter_procs_time, i_epoch, i_train_batch, visuals_flag):
        errors = self._model.get_current_errors()
        self._tb_visualizer.print_current_train_errors(i_epoch, i_train_batch, self._iters_per_epoch, errors,
                                                       iter_read_time, iter_procs_time, visuals_flag)
        self._epoch_train_e = append_dictionaries(self._epoch_train_e, errors)

    def _display_visualizer_train(self, total_steps, iter_read_time, iter_procs_time):
        self._tb_visualizer.display_current_results(self._model.get_current_visuals(), total_steps, is_train=True)
        self._tb_visualizer.plot_scalars(self._model.get_current_errors(), total_steps, is_train=True)
        self._tb_visualizer.plot_scalars(self._model.get_current_scalars(), total_steps, is_train=True)
        self._tb_visualizer.plot_time(iter_read_time, iter_procs_time, total_steps)
        self._tb_visualizer.plot_histograms(self._model.get_current_histograms(), total_steps, is_train=True)

    def _display_visualizer_avg_epoch(self, epoch):
        e_train = mean_dictionary(self._epoch_train_e)
        e_val = mean_dictionary(self._epoch_val_e)
        self._tb_visualizer.print_epoch_avg_errors(epoch, e_train, is_train=True)
        self._tb_visualizer.print_epoch_avg_errors(epoch, e_val, is_train=False)
        self._tb_visualizer.plot_scalars(e_train, epoch, is_train=True, is_mean=True)
        self._tb_visualizer.plot_scalars(e_val, epoch, is_train=False, is_mean=True)

    def _display_visualizer_val(self, i_epoch, total_steps):
        val_start_time = time.time()

        # set model to eval
        self._model.set_eval()

        # evaluate self._opt.num_iters_validate epochs
        val_errors = dict()
        with torch.no_grad():
            vis_batch_idx = np.random.randint(min(self._num_iters_validate, self._dataset_val_size))
            for i_val_batch, val_batch in enumerate(self._dataset_val):
                if i_val_batch == self._num_iters_validate:
                    break

                # evaluate model
                keep_data_for_visuals = (i_val_batch == vis_batch_idx)
                self._model.set_input(val_batch)
                self._model.forward(keep_data_for_visuals=keep_data_for_visuals)

                # store errors
                errors = self._model.get_current_errors()
                val_errors = append_dictionaries(val_errors, errors)

                # keep visuals
                if keep_data_for_visuals:
                    self._tb_visualizer.display_current_results(self._model.get_current_visuals(), total_steps,
                                                                is_train=False)
                    self._tb_visualizer.plot_histograms(self._model.get_current_histograms(), total_steps,
                                                        is_train=False)

            # store error
            val_errors = mean_dictionary(val_errors)
            self._epoch_val_e = append_dictionaries(self._epoch_val_e, val_errors)

        # visualize
        t = (time.time() - val_start_time)
        self._tb_visualizer.print_current_validate_errors(i_epoch, val_errors, t)
        self._tb_visualizer.plot_scalars(val_errors, total_steps, is_train=False)

        # set model back to train
        self._model.set_train()


if __name__ == "__main__":
    Train()
