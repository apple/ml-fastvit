#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All rights reserved.
#

import math


class CosineWDSchedule:
    def __init__(self, optimizer, t_max, eta_min=0, last_epoch=-1):
        self.last_epoch = last_epoch
        self.base_wds = [group["weight_decay"] for group in optimizer.param_groups]
        self.t_max = t_max
        self.eta_min = eta_min

    def _get_wd(self, optimizer):
        if self.last_epoch == 0:
            return self.base_wds
        elif (self.last_epoch - 1 - self.t_max) % (2 * self.t_max) == 0:
            return [
                group["weight_decay"]
                + (base_lr - self.eta_min) * (1 - math.cos(math.pi / self.t_max)) / 2
                for base_lr, group in zip(self.base_wds, optimizer.param_groups)
            ]
        return [
            (1 + math.cos(math.pi * self.last_epoch / self.t_max))
            / (1 + math.cos(math.pi * (self.last_epoch - 1) / self.t_max))
            * (group["weight_decay"] - self.eta_min)
            + self.eta_min
            for group in optimizer.param_groups
        ]

    def update_weight_decay(self, optimizer):
        self.last_epoch += 1
        values = self._get_wd(optimizer)
        for i, data in enumerate(zip(optimizer.param_groups, values)):
            param_group, wd = data
            # Avoid updating weight decay of param_groups that should not be decayed.
            if param_group["weight_decay"] > 0.0:
                param_group["weight_decay"] = wd
