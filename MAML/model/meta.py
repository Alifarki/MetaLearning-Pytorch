import  torch
from    torch import nn
from    torch import optim
from    torch.nn import functional as F
from    torch.utils.data import TensorDataset, DataLoader
from    torch import optim
import  numpy as np

from    .learner import Learner
# import Learner
from    copy import deepcopy


class Meta(nn.Module):

    def __init__(self, args, config):
        super(Meta, self).__init__()
        pass

    def forward(self, x_spt, y_spt, x_qry, y_qry):
        pass