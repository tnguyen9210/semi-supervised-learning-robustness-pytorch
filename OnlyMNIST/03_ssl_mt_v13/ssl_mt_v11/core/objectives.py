
import torch
import torch.nn as nn
import torch.nn.functional as F


class MT(nn.Module):
    def __init__(self, net, ema_factor):
        super().__init__()
        self.net = net
        self.net.train()
        self.ema_factor = ema_factor
        self.global_step = 0

    def forward(self, net, x):
        self.global_step += 1
        t_logit = self.net(x)
        net.update_batch_stats(False)
        s_logit = net(x) # recompute y since y as input of forward function is detached
        net.update_batch_stats(True)
        return (F.mse_loss(s_logit.softmax(1), t_logit.softmax(1).detach(), reduction="none").mean(1)).mean()

    def moving_average(self, parameters):
        ema_factor = min(1 - 1 / (self.global_step+1), self.ema_factor)
        for emp_p, p in zip(self.net.parameters(), parameters):
            emp_p.data = ema_factor * emp_p.data + (1 - ema_factor) * p.data
