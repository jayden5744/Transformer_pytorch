class InverseSqrt:
    """
    During warmup::
      lrs = torch.linspace(args.warmup_init_lr, args.lr, args.warmup_updates)
      lr = lrs[update_num]

    After warmup::
      decay_factor = args.lr * sqrt(args.warmup_updates)
      lr = decay_factor / sqrt(step)
    """

    def __init__(self, warmup, optimizer, warmup_init_lr: float = 1e-07, warmup_end_lr: float = 0.0005):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.warmup_init_lr = warmup_init_lr
        self.lr_step = (warmup_end_lr - warmup_init_lr) / warmup
        self.decay_factor = warmup_end_lr * warmup ** 0.5

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self.optimizer.step()

    def rate(self):
        "Implement `lrate` above"
        step = self._step
        if step < self.warmup:
            return self.warmup_init_lr + step * self.lr_step
        else:
            return self.decay_factor * step ** (-0.5)

    def zero_grad(self):
        return self.optimizer.zero_grad()
