import torch
from torch.optim import Adam

class ProjectedAdam(Adam):
    def __init__(self, params, projection_cache_map, lr=1e-3, betas=(0.9, 0.999), 
                 eps=1e-8, weight_decay=0, amsgrad=False):
        """
        Args:
            params: Iterable of parameters to optimize or dicts defining parameter groups.
            U_A (torch.Tensor): Left projection matrix.
            U_B (torch.Tensor): Right projection matrix.
            M (torch.Tensor): Mask matrix.
            ... (other args same as Adam)
        """
        defaults = dict(projection_cache_map=projection_cache_map)
        
        super().__init__(params, lr=lr, betas=betas, eps=eps, 
                         weight_decay=weight_decay, amsgrad=amsgrad)
        
        for group in self.param_groups:
            group.update(defaults)

    def reset_cache_old(self, new_projection_cache_map):
        """
        Resets the projection cache with a new one.
        Args:
            new_projection_cache_map (dict): New mapping of parameters to their projection caches.
        """
        defaults = dict(projection_cache_map=new_projection_cache_map)
        for group in self.param_groups:
            group.update(defaults)

    def reset_cache(self, new_projection_cache_map):
        defaults = dict(projection_cache_map=new_projection_cache_map)
        for group in self.param_groups:
            group.update(defaults)
            
            for p in group['params']:
                if p not in self.state: continue
                
                if p not in new_projection_cache_map: continue
                cache = new_projection_cache_map[p]
                
                U_A = cache['Ua'].to(device=p.device, dtype=p.dtype)
                U_B = cache['Ub'].to(device=p.device, dtype=p.dtype)
                M   = cache['M'].to(device=p.device, dtype=p.dtype)

                state = self.state[p]
                if 'exp_avg' in state:
                    m = state['exp_avg']
                    if m.ndim == 2:
                        # Apply projection to the momentum buffer
                        m_proj = U_B @ ( (U_B.T @ m @ U_A) * M.T ) @ U_A.T
                        m.copy_(m_proj)

    def reset_additional_cache(self, additional_projection_cache_map):
        defaults = dict(additional_projection_cache_map=additional_projection_cache_map)
        for group in self.param_groups:
            group.update(defaults)
            
            for p in group['params']:
                if p not in self.state: continue
                
                if p not in additional_projection_cache_map: continue
                cache = additional_projection_cache_map[p]
                
                U_A = cache['Ua'].to(device=p.device, dtype=p.dtype)
                U_B = cache['Ub'].to(device=p.device, dtype=p.dtype)
                M   = cache['M'].to(device=p.device, dtype=p.dtype)

                state = self.state[p]
                if 'exp_avg' in state:
                    m = state['exp_avg']
                    if m.ndim == 2:
                        m_proj = U_B @ ( (U_B.T @ m @ U_A) * M.T ) @ U_A.T
                        m.copy_(m_proj)
            
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                
                if grad.ndim != 2:
                    continue
                
                U_A = group['projection_cache_map'][p]['Ua'].to(device=grad.device, dtype=grad.dtype)
                U_B = group['projection_cache_map'][p]['Ub'].to(device=grad.device, dtype=grad.dtype)
                M = group['projection_cache_map'][p]['M'].to(device=grad.device, dtype=grad.dtype)
                grad_proj = U_B @ ( (U_B.T @ grad @ U_A) * M.T ) @ U_A.T
                
                if 'additional_projection_cache_map' in group and p in group['additional_projection_cache_map']:
                    U_A = group['additional_projection_cache_map'][p]['Ua'].to(device=grad.device, dtype=grad.dtype)
                    U_B = group['additional_projection_cache_map'][p]['Ub'].to(device=grad.device, dtype=grad.dtype)
                    M = group['additional_projection_cache_map'][p]['M'].to(device=grad.device, dtype=grad.dtype)
                    grad_proj = U_B @ ( (U_B.T @ grad_proj @ U_A) * M.T ) @ U_A.T

                p.grad.copy_(grad_proj)

        return super().step(closure)