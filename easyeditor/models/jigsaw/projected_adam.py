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
                grad_proj = U_B @ ( (U_B.T @ grad @ U_A) * M.T ) @ U_A.T

                # lamb = 500
                # A_inv = group['projection_cache_map'][p]['A_inv'].to(device=grad.device) * (1/lamb)
                # B_inv = group['projection_cache_map'][p]['B_inv'].to(device=grad.device) * (1/lamb)
                # grad_proj = A_inv @ grad @ B_inv

                p.grad.copy_(grad_proj)

        # --- Standard Adam Step ---
        # Now that p.grad is modified, Adam will use grad_proj for 
        # momentum and weight updates.
        return super().step(closure)

import math
import torch
from torch.optim import Adam

class ProjectedAdamOld(Adam):
    def __init__(self, params, projection_cache_map, lr=1e-3, betas=(0.9, 0.999), 
                 eps=1e-8, weight_decay=0, amsgrad=False):
        
        # Pass cache to defaults so it's available in param_groups
        defaults = dict(projection_cache_map=projection_cache_map)
        
        super().__init__(params, lr=lr, betas=betas, eps=eps, 
                         weight_decay=weight_decay, amsgrad=amsgrad)
        
        # Ensure defaults are set
        for group in self.param_groups:
            group.update(defaults)

        # Initial cache setup (move to device immediately)
        if projection_cache_map is not None:
            self.reset_cache(projection_cache_map)

    def reset_cache(self, new_projection_cache_map):
        """
        Updates the projection cache and pre-moves matrices to the correct device
        to avoid bottlenecking the step() loop.
        """
        # Update the mapping in the param groups
        defaults = dict(projection_cache_map=new_projection_cache_map)
        for group in self.param_groups:
            group.update(defaults)
            
            for p in group['params']:
                if new_projection_cache_map is not None and p not in new_projection_cache_map: 
                    continue
                
                # 1. OPTIMIZATION: Move matrices to GPU ONCE here, not in step()
                cache = new_projection_cache_map[p]
                # Modifying the dictionary in-place to store the GPU tensors
                cache['Ua'] = cache['Ua'].to(device=p.device, dtype=p.dtype)
                cache['Ub'] = cache['Ub'].to(device=p.device, dtype=p.dtype)
                cache['M']  = cache['M'].to(device=p.device, dtype=p.dtype)

                # 2. Project existing momentum (Your original good idea)
                state = self.state[p]
                if 'exp_avg' in state:
                    m = state['exp_avg']
                    if m.ndim == 2:
                        U_A, U_B, M = cache['Ua'], cache['Ub'], cache['M']
                        m_proj = U_B @ ( (U_B.T @ m @ U_A) * M.T ) @ U_A.T
                        m.copy_(m_proj)
                if 'exp_avg_sq' in state:
                    # Resetting to zero forces Adam to go through a "warmup" 
                    # phase for the variance, preventing the 1/sqrt(tiny) explosion.
                    state['exp_avg_sq'].zero_()
                    # print(f"value before zeroing: {state['exp_avg_sq'].mean().item()}")
                    
                    # Optional: If you use AMSGrad, reset that too
                    if 'max_exp_avg_sq' in state:
                        # print(f"value before zeroing max_exp_avg_sq: {state['max_exp_avg_sq'].mean().item()}")
                        state['max_exp_avg_sq'].zero_()

    def reset_cache_list(self, new_projection_cache_map_list):
        """
        Updates the optimizer with a list of projection caches. 
        It moves all matrices to the correct device and projects the 
        current momentum (exp_avg) against ALL provided subspaces sequentially.
        """
        # 1. Update the param_groups to store the NEW list of maps
        #    (The step function will need to be updated to handle a list, see below)
        defaults = dict(projection_cache_map_list=new_projection_cache_map_list)
        for group in self.param_groups:
            group.update(defaults)

            for p in group['params']:
                # Ensure we have state for this param before trying to project momentum
                state = self.state[p]
                
                # Iterate through every cache map in the list
                for cache_map in new_projection_cache_map_list:
                    
                    if p not in cache_map:
                        continue

                    cache = cache_map[p]
                    
                    # A. Move matrices to GPU immediately (Optimization)
                    # We check if they are already on the right device to avoid redundant copies
                    if cache['Ua'].device != p.device:
                        cache['Ua'] = cache['Ua'].to(device=p.device, dtype=p.dtype)
                        cache['Ub'] = cache['Ub'].to(device=p.device, dtype=p.dtype)
                        cache['M']  = cache['M'].to(device=p.device, dtype=p.dtype)

                    # B. Project existing momentum (exp_avg)
                    # We project m against this specific subspace. 
                    # Since we loop, we project against Subspace 1, then Subspace 2, etc.
                    if 'exp_avg' in state:
                        m = state['exp_avg']
                        if m.ndim == 2:
                            U_A, U_B, M = cache['Ua'], cache['Ub'], cache['M']
                            # Project m
                            m_proj = U_B @ ( (U_B.T @ m @ U_A) * M.T ) @ U_A.T
                            m.copy_(m_proj)
                    if 'exp_avg_sq' in state:
                        state['exp_avg_sq'].zero_()
                        state['step'] = 0  # Reset step to force Adam warmup
                        
                        if 'max_exp_avg_sq' in state:
                            state['max_exp_avg_sq'].zero_()

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step with 'Post-Adam' Projection.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            
            # Standard Adam Hyperparams
            beta1, beta2 = group['betas']
            weight_decay = group['weight_decay']
            eps = group['eps']
            lr = group['lr']
            amsgrad = group['amsgrad']

            for p in group['params']:
                if p.grad is None:
                    continue
                
                # --- 1. Standard Adam Logic (Manually Implemented) ---
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('ProjectedAdam does not support sparse gradients')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if amsgrad:
                        state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']

                state['step'] += 1

                # Weight decay
                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(eps)
                else:
                    denom = exp_avg_sq.sqrt().add_(eps)

                # Compute step size
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = lr * math.sqrt(bias_correction2) / bias_correction1
                
                # --- 2. Calculate the "Proposed" Update ---
                # This is the vector Adam *wants* to subtract from the weights
                # update = step_size * (exp_avg / denom)
                update = (exp_avg / denom).mul_(step_size)

                # --- 3. PROJECTION FIX: Project the Update, not the Gradient ---
                if group['projection_cache_map'] is not None and (p in group['projection_cache_map']) and (update.ndim == 2):
                    cache = group['projection_cache_map'][p]
                    # Matrices are already on device thanks to reset_cache fix
                    U_A, U_B, M = cache['Ua'], cache['Ub'], cache['M']
                    
                    # Apply projection to the calculated Adam update
                    # Formula: P(u) = U_A * ((U_A^T * u * U_B) * M) * U_B^T
                    update = U_B @ ( (U_B.T @ update @ U_A) * M.T ) @ U_A.T

                # --- 4. Apply Update ---
                p.add_(-update)

        return loss