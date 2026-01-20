import torch
from torch.optim import Optimizer

class ProjectedSGD(Optimizer):
    def __init__(self, params, projection_cache_map, lr=0.001):
        defaults = dict(lr=lr, projection_cache_map=projection_cache_map)
        super(ProjectedSGD, self).__init__(params, defaults)

    def reset_cache(self, new_projection_cache_map):
        defaults = dict(projection_cache_map=new_projection_cache_map)
        for group in self.param_groups:
            group.update(defaults)
    
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                
                # don't project the bias
                if grad.ndim != 2:
                    continue
                
                U_A = group['projection_cache_map'][p]['Ua'].to(device=grad.device, dtype=grad.dtype)
                U_B = group['projection_cache_map'][p]['Ub'].to(device=grad.device, dtype=grad.dtype)
                M = group['projection_cache_map'][p]['M'].to(device=grad.device, dtype=grad.dtype)
                grad_proj = U_A @ ( (U_A.T @ grad @ U_B) * M ) @ U_B.T

                p.add_(grad_proj, alpha=-lr)

        return loss