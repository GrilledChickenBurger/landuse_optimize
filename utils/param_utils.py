import torch


# utils/param_utils.py

class RunningStats:
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0

    def update(self, x: torch.Tensor, mask: torch.Tensor = None):
        """
        x: 任意形状 tensor
        mask: 与 x broadcastable，True 表示有效像素
        """
        if mask is not None:
            x = x[mask]

        if x.numel() == 0:
            return

        x = x.reshape(-1)

        for v in x:
            v = float(v)
            self.n += 1
            delta = v - self.mean
            self.mean += delta / self.n
            delta2 = v - self.mean
            self.M2 += delta * delta2

    def finalize(self):
        if self.n <= 1:
            return self.mean, 0.0
        var = self.M2 / (self.n - 1)
        return self.mean, var ** 0.5

