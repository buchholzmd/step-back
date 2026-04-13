"""
Author: Fabian Schaipp

Adapted from https://github.com/fabian-sp/ProxSPS/blob/main/sps/sps.py.

Main changes:
    * use .data in all computations
    * rename 'fstar' to 'lb'
"""

import torch
import warnings

from ..types import Params, LossClosure, OptFloat

class SPS(torch.optim.Optimizer):
    def __init__(self, 
                 params: Params, 
                 lr: float=1e-3,
                 weight_decay: float=0, 
                 lb: float=0, 
                 prox: bool=True)-> None:
        """
        
        Parameters
        ----------
        params : 
            PyTorch model parameters.
        lr : float, optional
            Learning rate. The default is 1e-3.
        weight_decay : float, optional
            Weigt decay parameter. The default is 0.
            If specified, the term weight_decay/2 * ||w||^2 is added to objective, where w are all model weights.
        fstar : float, optional
            Lower bound of loss function. The default is 0 (which is a lower bound for most loss functions).
        prox: bool, optional
            Whether to use ProxSPS or SPS.
            
        """
        
        params = list(params)
        defaults = dict(lr=lr, weight_decay=weight_decay)
        
        super(SPS, self).__init__(params, defaults)
        self.params = params
        
        self.lr = lr
        self.lb = lb
        self.prox = prox

        self.state['step_size_list'] = list()
        
        if len(self.param_groups) > 1:
            warnings.warn("More than one parameter group for SPS.")
        
        return
        
    def step(self, closure: LossClosure=None) -> OptFloat:
        """
        ProxSPS update

        See https://arxiv.org/abs/2301.04935.
        """
        
        with torch.enable_grad():
            loss = closure()
        
        # get lower bound of objective
        lb = self.lb
        
        # add l2-norm if not ProxSPS
        if not self.prox:
            r = 0          
            for group in self.param_groups:
                lmbda = group['weight_decay']
                for p in group['params']:
                    p.grad.add_(lmbda * p.data)  # gradients
                    r += (lmbda/2) * (p.data**2).sum() # loss
                    
            loss.add_(r)
        
                
        if self.prox:
            grad_norm, grad_dot_w = self.compute_grad_terms(need_gdotw=True)
        else:
            grad_norm, _ = self.compute_grad_terms(need_gdotw=False)
            
        ############################################################
        # update 
        for group in self.param_groups:
            lr = group['lr']
            lmbda = group['weight_decay']
            
            for p in group['params']:

                if self.prox:
                    nom = (1+lr*lmbda)*(loss - lb) - lr*lmbda*grad_dot_w
                else:
                    nom = loss - lb
                    
                denom = (grad_norm)**2 
                t1 = (nom/denom).item()
                t2 = max(0., t1)                 
                
                # compute tau^+
                tau = min(lr, t2) 
                
                p.data.add_(other=p.grad.data, alpha=-tau)
                if self.prox:
                    p.data.div_(1+lr*lmbda)
            
        ############################################################       
        # update state with metrics
        self.state['step_size_list'].append(t2) # works only if one param_group!

        return loss
    
    @torch.no_grad()
    def compute_grad_terms(self, need_gdotw=True):
        """
        computes:
            * norm of stochastic gradient ||grad||
            * inner product <grad,param> (needed only for prox=True). 
        """
        grad_norm = 0.
        grad_dot_w = 0.
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    raise KeyError("None gradient")
                
                g = p.grad.data
                grad_norm += torch.sum(torch.mul(g, g))
                if need_gdotw:
                    grad_dot_w += torch.sum(torch.mul(p.data, g))
          
        grad_norm = torch.sqrt(grad_norm)
        return grad_norm, grad_dot_w
    
class SGDScheduleFreeSPS(torch.optim.Optimizer):
    # TODO: implement this as a closure
    def __init__(self, params, 
                 lr: float=1e-3,
                 weight_decay: float=0, 
                 beta=0.9, 
                 ell_star=0.0, 
                 M=1.0):
        defaults = dict(lr=lr, weight_decay=weight_decay, beta=beta, ell_star=ell_star, M=M, train_mode=True)
        super().__init__(params, defaults)

        self.beta = beta
        self.ell_star = ell_star
        self.M = M
        self.k = 0

        self.ss = 0
        self.grad_norm = 0

        self.extra = []
        self.tru = 0
        self.denom = 0
        
        # Initialize polyak_events tracking
        self.state['polyak_events'] = []

    def eval(self):
        for group in self.param_groups:
            train_mode = group['train_mode']
            if train_mode:
                for p in group['params']:
                    state = self.state[p]
                    if 'z' in state:
                        # Set p.data to x
                        # Until now we are in train mode which means that p stores y, ie we can consider p=y
                        p.data.lerp_(end=state['z'], weight=1-1/self.beta) # p = 1/b*p+(1-1/b)*z
                        # x^t = 1/\b*y^t+(1-1/\b)*z
                group['train_mode'] = False

    def train(self):
        for group in self.param_groups:
            train_mode = group['train_mode']
            if not train_mode:
                for p in group['params']:
                    state = self.state[p]
                    if 'z' in state:
                        # Set p.data to y
                        # Until now we are in eval mode which means that p stores x, ie we can consider p=x
                        p.data.lerp_(end=state['z'], weight=1-self.beta) # p = b*p+(1-b)*z
                        # y^t = \b*x^t+(1-\b)*z^t
                group['train_mode'] = True

    def step(self, closure: LossClosure=None) -> OptFloat:
        with torch.enable_grad():
            loss = closure()
                
        ckp1 = 1/(self.k+1)
        self.k += 1

        _norm = 0.
        _dot = 0.

        for group in self.param_groups:
            if not group['train_mode']:
                raise Exception("Not in train mode!")
            
            for p in group['params']:
                if p.grad is None:
                    continue

                y = p.data # y = y^t
                grad = p.grad.data # grad = \nabla f_i(y^t)
                state = self.state[p]

                if 'z' not in state:
                    state['z'] = torch.clone(y)
                z = state['z']

                _dot += torch.sum(torch.mul(grad, z-p.data))
                _norm += torch.sum(torch.mul(grad, grad))
        
        self.grad_norm = _norm

        # Ensure M is a tensor for consistent comparison
        if not torch.is_tensor(self.M):
            self.M = torch.tensor(self.M, device=_norm.device)
        else:
            self.M = self.M.to(_norm.device)

        if self.M <= 0:
            sps = (max(loss.item()-self.ell_star+_dot, 0)/_norm).item()
            self.M = _norm
        else:
            sps = (max(loss.item()-self.ell_star+_dot, 0)/max(self.M, _norm)).item()
            self.denom = max(self.M, _norm).item()
            if max(self.M, _norm) == _norm:
                self.tru += 1
                self.denom = _norm.item()
            else:
                self.denom = self.M.item()
        
        self.ss = sps
        self.extra = [self.tru, self.denom]

        # Track polyak_events for logging
        if 'polyak_events' not in self.state:
            self.state['polyak_events'] = []
        
        self.state['polyak_events'].append({
            'step': self.k,
            'grad_norm': _norm.item(),
            'M': self.M.item() if torch.is_tensor(self.M) else self.M,
            'used_grad_norm':int((_norm > self.M).item()),
            'lr': sps
        })

        for group in self.param_groups:
            if not group['train_mode']:
                raise Exception("Not in train mode!")

            for p in group['params']:
                if p.grad is None:
                    continue

                y = p.data # y = y^t
                grad = p.grad.data # grad = \nabla f_i(y^t)
                state = self.state[p]

                if 'z' not in state:
                    state['z'] = torch.clone(y)
                z = state['z']

                if group['weight_decay'] != 0:
                    r = 0          
                    for group in self.param_groups:
                        lmbda = group['weight_decay']
                        for p in group['params']:
                            p.grad.add_(lmbda * p.data)  # gradients
                            r += (lmbda/2) * (p.data**2).sum() # loss
                            
                    loss.add_(r)

                # These operations update y in-place,
                # without computing x explicitly.
                y.lerp_(end=z, weight=ckp1) # y = (1-c_{t+1})y + c_{t+1}z
                y.add_(grad, alpha=sps*(self.beta*(1-ckp1)-1)) # y = y + \g*(\b(1-c_{t+1})-1)\nabla f_i(y^t)
                # y^{t+1} = (1-c_{t+1})y^t + c_{t+1}z^t + \g*(\b(1-c_{t+1})-1)\nabla f_i(y^t)

                # SGD step
                z.sub_(grad, alpha=sps) # z = z - \g\nabla f_i(y^t)
                # z^{t+1} = z^t - \g\nabla f_i(y^t)

        return loss