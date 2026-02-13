from torch.optim.lr_scheduler import LRScheduler
import math


class WarmupCosineLR(LRScheduler):
    def __init__(self, optimizer,
                 warmup_steps,
                 total_steps,
                 min_lr_ratio=0.1,
                 last_epoch=-1,
                 verbose=False):
        self.warmup_steps  = warmup_steps
        self.total_steps   = total_steps
        self.min_lr_ratio  = min_lr_ratio
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        step = self._step_count

        if step <= self.warmup_steps:
            scale = step / float(self.warmup_steps)
            return [base_lr * scale for base_lr in self.base_lrs]

        progress = (step - self.warmup_steps) / float(max(1, self.total_steps - self.warmup_steps))
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        min_lr_factor = self.min_lr_ratio
        scale = cosine_decay * (1 - min_lr_factor) + min_lr_factor
        return [base_lr * scale for base_lr in self.base_lrs]


class WarmupExponentialLR(LRScheduler):
    def __init__(self, optimizer, gamma, warmup_step, last_epoch = -1, verbose=False):
        self.gamma = gamma
        self.warmup_step = warmup_step
        super().__init__(optimizer, -1, verbose=verbose)

    def get_lr(self):
        if self._step_count < self.warmup_step:
            ratio = float(self._step_count) / float(self.warmup_step)
            return [group['initial_lr'] * ratio
                    for group in self.optimizer.param_groups]
        else:
            return [group['lr'] * self.gamma
                for group in self.optimizer.param_groups]
