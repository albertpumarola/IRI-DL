import torch
from collections import OrderedDict
from src.utils import util
from .models import BaseModel
from src.networks.networks import NetworksFactory
from src.utils.plots import plot_estim
import numpy as np

class Model1(BaseModel):
    def __init__(self, opt):
        super(Model1, self).__init__(opt)
        self._name = 'Model1'

        # init input params
        self._init_set_input_params()

        # create networks
        self._init_create_networks()

        # init train variables
        if self._is_train:
            self._init_train_vars()

        # load networks and optimizers
        if not self._is_train or self._opt["model"]["load_epoch"] > 0:
            self.load()

        # init losses
        if self._is_train:
            self._init_losses()

        # prefetch inputs
        self._init_prefetch_inputs()

    def _init_set_input_params(self):
        self._B = self._opt[self._dataset_type]["batch_size"]               # batch
        self._S = self._opt[self._dataset_type]["image_size"]               # image size
        self._Ci = self._opt[self._dataset_type]["img_nc"]                  # num channels image
        self._Ct = self._opt[self._dataset_type]["target_nc"] * self._B     # num channels target

    def _init_create_networks(self):
        # create reg
        reg_type = self._opt["networks"]["reg"]["type"]
        reg_hyper_params = self._opt["networks"]["reg"]["hyper_params"]
        self._reg = NetworksFactory.get_by_name(reg_type, **reg_hyper_params)
        self._reg = torch.nn.DataParallel(self._reg, device_ids=self._reg_gpus_ids)

    def _init_train_vars(self):
        self._current_lr = self._opt["train"]["reg_lr"]
        self._optimizer = torch.optim.SGD(self._reg.parameters(), lr=self._current_lr)

    def _init_losses(self):
        self._criterion = torch.nn.CrossEntropyLoss().to(self._device_master)

    def _init_prefetch_inputs(self):
        self._input_img = torch.zeros([self._B, self._Ci, self._S, self._S]).to(self._device_master)
        self._input_target = torch.zeros([self._Ct], dtype=torch.long).to(self._device_master)

    def set_input(self, input):
        # copy values
        self._input_img.copy_(input['img'])
        self._input_target.copy_(input['target'])

        # move to gpu
        self._input_img = self._input_img.to(self._device_master)
        self._input_target = self._input_target.to(self._device_master)

    def set_train(self):
        self._reg.train()
        self._is_train = True

    def set_eval(self):
        self._reg.eval()
        self._is_train = False

    def evaluate(self):
        # set model to eval
        is_train = self._is_train
        if is_train:
            self.set_eval()

        # estimate object categories
        with torch.no_grad():
            self.forward(keep_data_for_visuals=True, estimate_loss=False)
            eval = np.transpose(self._vis_input_img, (1, 2, 0))

        # set model back to train if necessary
        if is_train:
            self.set_train()

        return eval

    def optimize_parameters(self, keep_data_for_visuals=False):
        if self._is_train:

            # calculate loss
            loss = self.forward(keep_data_for_visuals=keep_data_for_visuals)

            # optimize
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()

        else:
            raise ValueError('Trying to optimize in non-training mode!')

    def forward(self, keep_data_for_visuals=False, estimate_loss=True):
        # generate img
        estim = self._estimate(self._input_img)

        # estimate loss
        if estimate_loss:
            self._loss_gt = self._criterion(estim, self._input_target)
            total_loss = self._loss_gt
        else:
            total_loss = -1

        # keep visuals
        if keep_data_for_visuals:
            self._keep_data(estim)

        return total_loss

    def _estimate(self, img):
        return self._reg.forward(img)

    def _keep_data(self, estim):
        predicted = estim.max(1)[1].detach().cpu().numpy()
        vis_img = util.tensor2im(self._input_img.detach(), unnormalize=True, to_numpy=True)
        self._vis_input_img = plot_estim(vis_img, predicted, self._input_target.detach().cpu().numpy())

    def get_image_paths(self):
        return OrderedDict()

    def get_current_errors(self):
        loss_dict = OrderedDict()
        loss_dict["loss_gt"] = self._loss_gt.item()
        return loss_dict

    def get_current_scalars(self):
        return OrderedDict([('lr', self._current_lr)])

    def get_current_visuals(self):
        visuals = OrderedDict()
        visuals["1_estim_img"] = self._vis_input_img
        return visuals

    def save(self, epoch_label, save_type, do_remove_prev=True):
        # save networks
        self._save_network(self._reg, 'nn_reg', epoch_label, save_type, do_remove_prev)
        self._save_optimizer(self._optimizer, 'o_reg', epoch_label, save_type, do_remove_prev)

    def load(self):
        # load networks
        load_epoch = self._opt["model"]["load_epoch"]
        self._load_network(self._reg, 'nn_reg', load_epoch)
        if self._is_train:
            self._load_optimizer(self._optimizer, "o_reg", load_epoch)

    def update_learning_rate(self, curr_epoch):
        initial_lr = float(self._opt["train"]["reg_lr"])
        nepochs_no_decay = self._opt["train"]["nepochs_no_decay"]
        nepochs_decay = self._opt["train"]["nepochs_decay"]

        # update lr
        if curr_epoch <= nepochs_no_decay:
            self._current_lr = initial_lr
        else:
            new_lr = self._lr_linear(self._current_lr, nepochs_decay, initial_lr)
            self._update_learning_rate(self._optimizer, "reg", self._current_lr, new_lr)
            self._current_lr = new_lr
