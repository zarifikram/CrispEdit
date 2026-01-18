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
        # We pass the projection matrices into the defaults so they are available 
        # in param_groups. This allows different U_A/U_B/M for different groups 
        # if you ever need that flexibility.
        defaults = dict(projection_cache_map=projection_cache_map)
        
        # Initialize the standard Adam optimizer
        super().__init__(params, lr=lr, betas=betas, eps=eps, 
                         weight_decay=weight_decay, amsgrad=amsgrad)
        
        # Update defaults with the projection matrices so they are stored in groups
        for group in self.param_groups:
            group.update(defaults)

    # def reset_cache_old(self, new_projection_cache_map):
    #     """
    #     Resets the projection cache with a new one.
    #     Args:
    #         new_projection_cache_map (dict): New mapping of parameters to their projection caches.
    #     """
    #     defaults = dict(projection_cache_map=new_projection_cache_map)
    #     for group in self.param_groups:
    #         group.update(defaults)

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
                        m_proj = U_A @ ( (U_A.T @ m @ U_B) * M ) @ U_B.T
                        m.copy_(m_proj)
            
    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:

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

                # lamb = 500
                # A_inv = group['projection_cache_map'][p]['A_inv'].to(device=grad.device) * (1/lamb)
                # B_inv = group['projection_cache_map'][p]['B_inv'].to(device=grad.device) * (1/lamb)
                # grad_proj = A_inv @ grad @ B_inv

                p.grad.copy_(grad_proj)

        # --- Standard Adam Step ---
        # Now that p.grad is modified, Adam will use grad_proj for 
        # momentum and weight updates.
        return super().step(closure)