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
                
                if not group['projection_cache_map'] or (p not in group['projection_cache_map']):
                    p.add_(grad, alpha=-lr)
                    continue
                U_A = group['projection_cache_map'][p]['Ua'].to(device=grad.device, dtype=grad.dtype)
                U_B = group['projection_cache_map'][p]['Ub'].to(device=grad.device, dtype=grad.dtype)
                M = group['projection_cache_map'][p]['M'].to(device=grad.device, dtype=grad.dtype)
                grad = U_B @ ( (U_B.T @ grad @ U_A) * M.T ) @ U_A.T

                p.add_(grad, alpha=-lr)
                # g_norm = grad.norm().item()
                # proj_norm = grad_proj.norm().item()
                # ratio = proj_norm / (g_norm + 1e-8)

                # print(f"Param shape: {grad.shape}")
                # print(f"Original Grad Norm: {g_norm:.6f}")
                # print(f"Projected Grad Norm: {proj_norm:.6f} (Ratio: {ratio:.4f})")
                # print(f"M max value: {M.max().item()}, M min value: {M.min().item()}")

        return loss