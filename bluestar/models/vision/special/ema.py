from torch import nn
from copy import deepcopy

class EMA_Model(nn.Module):
    def __init__(self, model, decay=0.9999, device=None):
        super(EMA_Model, self).__init__()

        self.ema_model = deepcopy(model)
        self.ema_model.eval()
        self.decay = decay
        self.device = device
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.ema_model.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        # update ema_model with exponential moving average
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        # map ema_model to model
        self._update(model, update_fn=lambda e, m: m)