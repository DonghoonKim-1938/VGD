import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa


class MyBatchNorm2d(nn.BatchNorm2d):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # self._check_input_dim(x)

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # calculate running estimates
        if self.training:
            # calc mean in train batch
            mean = x.mean([0, 2, 3])

            # use biased var in train batch
            var = x.var([0, 2, 3], unbiased=False)
            n = x.numel() / x.size(1)

            with torch.no_grad():

                self.running_mean = exponential_average_factor * mean\
                    + (1 - exponential_average_factor) * self.running_mean

                # update running_var with unbiased var
                self.running_var = exponential_average_factor * var * n / (n - 1)\
                    + (1 - exponential_average_factor) * self.running_var
        else:
            mean = self.running_mean
            var = self.running_var

        x = (x - mean[None, :, None, None]) / (torch.sqrt(var[None, :, None, None] + self.eps))

        if self.affine:
            x = x * self.weight[None, :, None, None] + self.bias[None, :, None, None]

        return x
