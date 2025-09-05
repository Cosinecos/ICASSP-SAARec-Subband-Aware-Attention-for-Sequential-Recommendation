import math
class CosineWithWarmup:
    def __init__(self, optimizer, total_epochs, warmup_epochs=0, base_lr=1e-3):
        self.opt = optimizer; self.total = total_epochs; self.warm = warmup_epochs
        self.base_lr = base_lr; self.cur_epoch = 0
    def step(self):
        self.cur_epoch += 1
        if self.cur_epoch <= self.warm:
            lr = self.base_lr * self.cur_epoch / max(1, self.warm)
        else:
            t = (self.cur_epoch - self.warm) / max(1, self.total - self.warm)
            lr = 0.5 * self.base_lr * (1 + math.cos(math.pi * t))
        for g in self.opt.param_groups: g['lr'] = lr
        return lr
