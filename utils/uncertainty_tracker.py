import torch

class EMATracker:
    def __init__(self, alpha=0.1, device='cpu'):
        self.alpha = alpha
        self.ema_values = {}
        self.device = device  # Puedes poner 'cuda' si realmente quieres mantenerlo en GPU

    @torch.no_grad()
    def update(self, key, value):
        # Si es tensor, moverlo temporalmente al dispositivo del EMA
        if isinstance(value, torch.Tensor):
            v = value.detach().to(self.device)
        else:
            v = torch.tensor(value, device=self.device)

        if key not in self.ema_values:
            # Clonamos sin gradiente
            self.ema_values[key] = v.clone()
        else:
            # Operaciones in-place sin crear tensores intermedios
            self.ema_values[key].mul_(1 - self.alpha).add_(v, alpha=self.alpha)

        return self.ema_values[key]

    def get(self, key):
        return self.ema_values.get(key, None)