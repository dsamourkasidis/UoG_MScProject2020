import torch
from torch import nn
import torch.nn.functional as functional
from torch.autograd import Function
import numpy as np

import sys

sys.path.append("../../")
from deq_modules.deq import *


class CpNltNetDEQModule(DEQModule):
    """ See DEQModule class for documentation """

    def __init__(self, func, func_copy):
        super(CpNltNetDEQModule, self).__init__(func, func_copy)

    def forward(self, z1s, us, z0, **kwargs):
        z1s = z1s.permute(0, 2, 1)
        bsz, total_hsize, seq_len = z1s.size()
        train_step = kwargs.get('train_step', -1)
        subseq_len = kwargs.get('subseq_len', seq_len)
        threshold = kwargs.get('threshold', 50)

        #if us is None:
         #   raise ValueError("Input injection is required.")

        # Use this line for longer sequences:
        #     self._solve_by_subseq(z1s, us, z0, threshold, train_step, subseq_len=subseq_len)

        # Use these lines for shorter sequences:
        z1s_out = RootFind.apply(self.func, z1s, us, z0, threshold, train_step)
        if self.training:
            z1s_out = RootFind.f(self.func, z1s_out, us, z0, threshold, train_step)
            z1s_out = self.Backward.apply(self.func_copy, z1s_out, us, z0, threshold, train_step)
        return z1s_out
