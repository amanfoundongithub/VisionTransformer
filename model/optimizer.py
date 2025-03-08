import torch
from torch.optim import Adam

class TransformerOptimizer:
    def __init__(self, model, warmup_steps):
        self.optimizer = Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-9)
        self.d_model = model.embed_dim
        self.warmup_steps = warmup_steps
        self._step = 0
        self._rate = 0

    def step(self):
        """Performs a single optimization step with custom learning rate scheduling."""
        self._step += 1
        rate = self._get_lr()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def _get_lr(self):
        """Calculates the learning rate according to the Transformer schedule."""
        return (self.d_model ** -0.5) * min(self._step ** -0.5, self._step * (self.warmup_steps ** -1.5))

    def zero_grad(self):
        self.optimizer.zero_grad()