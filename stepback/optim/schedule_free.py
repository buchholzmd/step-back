"""
Borrowed from: https://github.com/facebookresearch/schedule_free/blob/main/schedulefree/sgd_schedulefree.py
          and  https://github.com/facebookresearch/schedule_free/blob/main/schedulefree/adamw_schedulefree.py
"""
from typing import Tuple, Union, Optional, Iterable, Dict, Callable, Any
from typing_extensions import TypeAlias
import torch
import torch.optim
try:
    from torch.optim.optimizer import ParamsT
except ImportError:
    ParamsT : TypeAlias = Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]]

import math

class SGDScheduleFree(torch.optim.Optimizer):
    r"""
    Schedule-Free SGD
    As the name suggests, no scheduler is needed with this optimizer. 
    To add warmup, rather than using a learning rate schedule you can just
    set the warmup_steps parameter.

    This optimizer requires that .train() and .eval() be called before the
    beginning of training and evaluation respectively. The optimizer should
    also be placed in eval mode when saving checkpoints.
    
    Arguments:
        params (iterable): 
            Iterable of parameters to optimize or dicts defining 
            parameter groups.
        lr (float): 
            Learning rate parameter (default 1.0)
        momentum (float): momentum factor, must be between 0 and 1 exclusive
            (default: 0.9)
        eps (float): 
            Term added to the denominator outside of the root operation to 
            improve numerical stability. (default: 1e-8).
        weight_decay (float): 
            Weight decay, i.e. a L2 penalty (default: 0).
        warmup_steps (int): Enables a linear learning rate warmup (default 0).
        r (float): Use polynomial weighting in the average 
            with power r (default 0).
        weight_lr_power (float): During warmup, the weights in the average will
            be equal to lr raised to this power. Set to 0 for no weighting
            (default 2.0).
        foreach (bool): Use a foreach-backed implementation of the optimizer.
            Should be significantly faster, but will have higher peak memory
            usage (default True if supported in your PyTorch version).
        mode (String): Determines the setting of the weights (c_t's) based on
            the original paper ("schedule-free"), 
            theory ("schedulet"), or 
            Polyak-Rupert averaging ("pr-avg")
    """
    def __init__(self,
                 params: ParamsT,
                 lr: Union[float, torch.Tensor] = 1.0,
                 momentum: float = 0.9,
                 eps=1e-8,
                 weight_decay: float = 0,
                 warmup_steps: int = 0,
                 r: float = 0.0,
                 weight_lr_power: float = 2,
                 M=1.0,
                 polyak_lambda=1.0,
                 polyak_lb=0.0,
                 foreach: Optional[bool] = hasattr(torch, "_foreach_mul_"),
                 mode: str = "schedule-free"
                 ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if momentum <= 0 or momentum >= 1:
            raise ValueError("Momentum must be between 0 and 1 exclusive: {}".format(momentum))

        defaults = dict(lr=lr,
                        momentum=momentum,
                        eps=eps,
                        r=r,
                        k=0,
                        warmup_steps=warmup_steps,
                        train_mode=False,
                        weight_sum=0.0,
                        lr_max=-1.0,
                        scheduled_lr=0.0,
                        weight_lr_power=weight_lr_power,
                        weight_decay=weight_decay,
                        M=M,
                        polyak_lambda=polyak_lambda,
                        polyak_lb=polyak_lb,
                        foreach=foreach,
                        mode=mode)
        super().__init__(params, defaults)
        # Signal to train.py that this optimizer self-regulates step size;
        # gradient clipping should be skipped to avoid corrupting the gradient norm.
        self._uses_polyak_step = 'polyak' in mode
        # Signal to train.py that loss must be passed to step() (Polyak step needs it).
        self._pass_loss = 'polyak' in mode
    
    @torch.no_grad()
    def eval(self):
        for group in self.param_groups:
            train_mode = group['train_mode']
            momentum = group['momentum']
            if train_mode:
                for p in group['params']:
                    state = self.state[p]
                    if 'z' in state:
                        # Set p to x
                        p.lerp_(end=state['z'].to(p.device), weight=1-1/momentum)
                group['train_mode'] = False

    @torch.no_grad()
    def train(self):
        for group in self.param_groups:
            train_mode = group['train_mode']
            momentum = group['momentum']
            if not train_mode:
                for p in group['params']:
                    state = self.state[p]
                    if 'z' in state:
                        # Set p to y
                        p.lerp_(end=state['z'].to(p.device), weight=1-momentum)
                group['train_mode'] = True

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None, loss=None) -> Optional[float]:
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
            loss (tensor, optional): Pre-computed loss value. Used directly if
                no closure is provided (e.g. when backward() was already called).
        """
        if not self.param_groups[0]['train_mode']:
            raise Exception("Optimizer was not in train mode when step is called. "
                            "Please insert .train() and .eval() calls on the "
                            "optimizer. See documentation for details.")

        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            momentum = group['momentum']
            lr = group['lr']
            eps = torch.tensor(group['eps'])
            weight_decay = group['weight_decay']
            k = group['k']
            r = group['r']
            warmup_steps = group['warmup_steps']            
            weight_lr_power = group['weight_lr_power']

            if k < warmup_steps:
              sched = (k+1) / warmup_steps
            else:
              sched = 1.0
            lr = group['lr']*sched
            group['scheduled_lr'] = lr # For logging purposes
            
            lr_max = group['lr_max'] = max(lr, group['lr_max'])

            assert(group['mode'] in ['schedule-free', 
                                     'schedulet', 
                                     'pr-avg',
                                     'schedule-free-polyak'])
            
            if group['mode'] == 'schedule-free':
                weight = ((k+1)**r) * (lr_max**weight_lr_power)
                weight_sum = group['weight_sum'] = group['weight_sum'] + weight

                try:
                    ckp1 = weight/weight_sum
                except ZeroDivisionError:
                    ckp1 = 0
            elif 'schedulet' in group['mode']:
                weight = lr
                weight_sum = group['weight_sum'] = group['weight_sum'] + weight

                try:
                    ckp1 = weight/weight_sum
                except ZeroDivisionError:
                    ckp1 = 0
            elif 'pr-avg' in group['mode'] or 'polyak' in group['mode']:
                ckp1 = 1/(k+1)

            active_p = [p for p in group['params'] if p.grad is not None]

            for p in active_p:
                if 'z' not in self.state[p]:
                    self.state[p]['z'] = torch.clone(p, memory_format=torch.preserve_format)

            if group['foreach'] and len(active_p) > 0:
                y, grad, z = zip(*[(p, p.grad, self.state[p]['z']) 
                                for p in active_p])

                if 'polyak' in group['mode']:
                    # compute averaged grads
                    grad_norm = sum((g**2).sum() for g in grad)
                    if not torch.is_tensor(group['M']):
                        M = group['M'] = torch.tensor([group['M']]).to(grad_norm.device)
                    else:
                        M = group['M']

                    # Only update M if polyak_lambda > 0 (keep M constant when polyak_lambda = 0)
                    if group['polyak_lambda'] is not None and group['polyak_lambda'] > 0:
                        M = group['M'] = group['polyak_lambda'] * group['M'] + (1-group['polyak_lambda']) * grad_norm
                    if M <= 0:
                        M = group['M'] = grad_norm
                    
                    loss_gap = loss - group['polyak_lb']
                    proj_gap = sum(torch.sum(g * (zi - yi)) for g, zi, yi in zip(grad, z, y))
                    numerator = torch.clamp(loss_gap + proj_gap, min=0)
                    denominator = torch.max(grad_norm, M).item()
                    lr = (numerator / denominator).item() if numerator > 0 else 0.0

                    group['lr'] = float(lr)
                    
                    ckp1 = 1/(k+1)

                    if 'polyak_events' not in self.state:
                        self.state['polyak_events'] = []

                    self.state['polyak_events'].append({
                                'step': group['k'],
                                'grad_norm': grad_norm.item(),
                                'M': M.item(),
                                'used_grad_norm': int((grad_norm > M).item()),
                                'lr': group['lr']
                    })
                
                # Apply weight decay
                if weight_decay != 0:
                    torch._foreach_add_(grad, y, alpha=weight_decay)

                # Check for NaN or Inf in gradients
                if any(torch.isnan(g).any() or torch.isinf(g).any() for g in grad):
                    raise RuntimeError("NaN or Inf detected in gradients.")

                # These operations update y in-place,
                # without computing x explicitly.
                torch._foreach_lerp_(y, z, weight=ckp1)
                torch._foreach_add_(y, grad, alpha=lr*(momentum*(1-ckp1)-1))

                # SGD step
                torch._foreach_sub_(z, grad, alpha=lr)
            else:
                # For polyak mode, compute lr once across all parameters BEFORE the parameter loop
                if 'polyak' in group['mode']:
                    # Compute aggregate grad_norm across all parameters
                    grad_norm = sum((p.grad**2).sum() for p in active_p)
                    if not torch.is_tensor(group['M']):
                        M = group['M'] = torch.tensor([group['M']]).to(grad_norm.device)
                    else:
                        M = group['M']

                    # Only update M if polyak_lambda > 0 (keep M constant when polyak_lambda = 0)
                    if group['polyak_lambda'] is not None and group['polyak_lambda'] > 0:
                        M = group['M'] = group['polyak_lambda'] * group['M'] + (1-group['polyak_lambda']) * grad_norm
                    if M <= 0:
                        M = group['M'] = grad_norm

                    loss_gap = loss - group['polyak_lb']
                    proj_gap = sum(torch.sum(p.grad * (self.state[p]['z'] - p)) for p in active_p)
                    # Compute lr once for all parameters
                    numerator = torch.clamp(loss_gap + proj_gap, min=0)
                    lr = (numerator / torch.max(grad_norm, M).item()).item() if numerator > 0 else 0.0

                    group['lr'] = lr

                    if 'polyak_events' not in self.state:
                        self.state['polyak_events'] = []

                    self.state['polyak_events'].append({
                            'step': group['k'],
                            'grad_norm': grad_norm.item(),
                            'M': M.item(),
                            'used_grad_norm': int((grad_norm > M).item()),
                            'lr': group['lr']
                    })

                for p in active_p:
                    y = p # Notation to match theory
                    grad = p.grad
                    z = self.state[p]['z']

                    # Apply weight decay
                    if weight_decay != 0:
                        grad.add_(y, alpha=weight_decay)

                    # These operations update y in-place,
                    # without computing x explicitly.
                    y.lerp_(end=z, weight=ckp1)
                    y.add_(grad, alpha=lr*(momentum*(1-ckp1)-1))

                    # SGD step
                    z.sub_(grad, alpha=lr)

            group['k'] = k+1
        return loss

class AdamWScheduleFree(torch.optim.Optimizer):
    r"""
    Schedule-Free AdamW
    As the name suggests, no scheduler is needed with this optimizer. 
    To add warmup, rather than using a learning rate schedule you can just
    set the warmup_steps parameter.
    
    This optimizer requires that .train() and .eval() be called before the
    beginning of training and evaluation respectively. The optimizer should
    also be placed in eval mode when saving checkpoints.
    
    Arguments:
        params (iterable): 
            Iterable of parameters to optimize or dicts defining 
            parameter groups.
        lr (float): 
            Learning rate parameter (default 0.0025)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999)).
        eps (float): 
            Term added to the denominator outside of the root operation to 
            improve numerical stability. (default: 1e-8).
        weight_decay (float): 
            Weight decay, i.e. a L2 penalty (default: 0).
        warmup_steps (int): Enables a linear learning rate warmup (default 0).
        r (float): Use polynomial weighting in the average 
            with power r (default 0).
        weight_lr_power (float): During warmup, the weights in the average will
            be equal to lr raised to this power. Set to 0 for no weighting
            (default 2.0).
        foreach (bool): Use a foreach-backed implementation of the optimizer.
            Should be significantly faster, but will have higher peak memory
            usage (default True if supported in your PyTorch version).
        mode (String): Determines the setting of the weights (c_t's) based on
            the original paper ("schedule-free"), 
            theory ("schedulet"), or 
            Polyak-Rupert averaging ("pr-avg")
    """
    def __init__(self,
                 params: ParamsT,
                 lr: Union[float, torch.Tensor] = 0.0025,
                 betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8,
                 weight_decay: float = 0,
                 warmup_steps: int = 0,
                 r: float = 0.0,
                 weight_lr_power: float = 2.0,
                 M=1.0,
                 polyak_lambda=1.0,
                 polyak_lb=0.0,
                 foreach: Optional[bool] = hasattr(torch, "_foreach_mul_"),
                 mode: str = "schedule-free"
                 ):

        defaults = dict(lr=lr, 
                        betas=betas, 
                        eps=eps,
                        r=r,
                        k=0,
                        warmup_steps=warmup_steps,
                        train_mode=False,
                        weight_sum=0.0,
                        lr_max=-1.0,
                        scheduled_lr=0.0,
                        weight_lr_power=weight_lr_power,
                        weight_decay=weight_decay,
                        M=M,
                        polyak_lambda=polyak_lambda,
                        polyak_lb=polyak_lb,
                        foreach=foreach,
                        mode=mode)
        super().__init__(params, defaults)
        # Signal to train.py that loss must be passed to step() in Polyak mode.
        self._pass_loss = 'polyak' in mode

    @torch.no_grad()
    def eval(self):
        for group in self.param_groups:
            train_mode = group['train_mode']
            beta1, _ = group['betas']
            if train_mode:
                for p in group['params']:
                    state = self.state[p]
                    if 'z' in state:
                        # Set p to x
                        p.lerp_(end=state['z'].to(p.device), weight=1-1/beta1)
                group['train_mode'] = False

    @torch.no_grad()
    def train(self):
        for group in self.param_groups:
            train_mode = group['train_mode']
            beta1, _ = group['betas']
            if not train_mode:
                for p in group['params']:
                    state = self.state[p]
                    if 'z' in state:
                        # Set p to y
                        p.lerp_(end=state['z'].to(p.device), weight=1-beta1)
                group['train_mode'] = True

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None, loss=None) -> Optional[float]:
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
            loss (tensor, optional): Pre-computed loss value. Used directly if
                no closure is provided (e.g. when backward() was already called).
        """
        if not self.param_groups[0]['train_mode']:
            raise Exception("Optimizer was not in train mode when step is called. "
                            "Please insert .train() and .eval() calls on the "
                            "optimizer. See documentation for details.")

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            eps = group['eps']
            beta1, beta2 = group['betas']
            decay = group['weight_decay']
            k = group['k']
            r = group['r']
            warmup_steps = group['warmup_steps']
            weight_lr_power = group['weight_lr_power']
            
            if k < warmup_steps:
              sched = (k+1) / warmup_steps
            else:
              sched = 1.0
            
            bias_correction2 = 1 - beta2 ** (k+1)
            lr = group['lr']*sched
            group['scheduled_lr'] = lr # For logging purposes
            
            lr_max = group['lr_max'] = max(lr, group['lr_max'])
            
            assert(group['mode'] in ['schedule-free-adam', 
                                     'schedulet-adam', 
                                     'pr-avg-adam',
                                     'schedule-free-polyak-adam'])
            
            if 'schedule-free' in group['mode']:
                weight = ((k+1)**r) * (lr_max**weight_lr_power)
                weight_sum = group['weight_sum'] = group['weight_sum'] + weight

                try:
                    ckp1 = weight/weight_sum
                except ZeroDivisionError:
                    ckp1 = 0
            elif 'schedulet' in group['mode']:
                weight = lr
                weight_sum = group['weight_sum'] = group['weight_sum'] + weight

                try:
                    ckp1 = weight/weight_sum
                except ZeroDivisionError:
                    ckp1 = 0
            elif 'pr-avg' in group['mode']:
                ckp1 = 1/(k+1)

            active_p = [p for p in group['params'] if p.grad is not None]
            
            for p in active_p:
                if 'z' not in self.state[p]:
                    self.state[p]['z'] = torch.clone(p, memory_format=torch.preserve_format)
                    self.state[p]['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

            if group['foreach'] and len(active_p) > 0:
                y, grad, exp_avg_sq, z = zip(*[(p,
                                                p.grad,
                                                self.state[p]['exp_avg_sq'],
                                                self.state[p]['z'])
                                                for p in active_p])

                # Decay the first and second moment running average coefficient
                torch._foreach_mul_(exp_avg_sq, beta2)
                torch._foreach_addcmul_(exp_avg_sq, grad, grad, value=1-beta2)
                denom = torch._foreach_div(exp_avg_sq, bias_correction2)
                torch._foreach_sqrt_(denom)
                torch._foreach_add_(denom, eps)

                if 'polyak' in group['mode']:
                    # proj_gap = <g_raw, z - y> must be computed with the RAW gradient
                    # (before Adam normalization). Formula: a_t = [f(y)-f* + <g,z-y>]+ / max(M, ||g~||^2)
                    # where g~ = D^{-1/2} g and ||g~||^2 = sum(g^2/D) (Remark C.1).
                    proj_gap = sum(torch.sum(g * (zi - yi)) for g, zi, yi in zip(grad, z, y))
                    # grad_norm = ||D^{-1/2}g||^2 = sum(g^2/D) = sum(g_raw * (g_raw/D))
                    # Must be computed BEFORE in-place normalization (needs both raw grad and D).
                    # Matches SCHEDULETA: _norm += torch.sum(grad.mul(grad.div(Dk)))
                    grad_normalized_for_norm = torch._foreach_div(grad, denom)  # g/D, new tensors
                    grad_norm = sum(torch.sum(g * gn) for g, gn in zip(grad, grad_normalized_for_norm))
                    del grad_normalized_for_norm

                # Normalize grad in-place for memory efficiency
                torch._foreach_div_(grad, denom)

                # Weight decay calculated at y
                if decay != 0:
                    torch._foreach_add_(grad, y, alpha=decay)

                if 'polyak' in group['mode']:
                    if not torch.is_tensor(group['M']):
                        M = group['M'] = torch.tensor([group['M']]).to(grad_norm.device)
                    else:
                        M = group['M']

                    # Only update M if polyak_lambda > 0 (keep M constant when polyak_lambda = 0)
                    if group['polyak_lambda'] is not None and group['polyak_lambda'] > 0:
                        M = group['M'] = group['polyak_lambda'] * group['M'] + (1-group['polyak_lambda']) * grad_norm
                    if M <= 0:
                        M = group['M'] = grad_norm

                    loss_gap = loss - group['polyak_lb']
                    numerator = torch.clamp(loss_gap + proj_gap, min=0)
                    denominator = torch.max(grad_norm, M).item()
                    lr = (numerator / denominator).item() if numerator > 0 else 0.0

                    group['lr'] = float(lr)
                    ckp1 = 1/(k+1)

                    if 'polyak_events' not in self.state:
                        self.state['polyak_events'] = []

                    self.state['polyak_events'].append({
                                'step': group['k'],
                                'grad_norm': grad_norm.item(),
                                'M': M.item(),
                                'used_grad_norm': int((grad_norm > M).item()),
                                'lr': group['lr']
                    })

                # These operations update y in-place,
                # without computing x explicitly.
                torch._foreach_lerp_(y, z, weight=ckp1)
                torch._foreach_add_(y, grad, alpha=lr*(beta1*(1-ckp1)-1))

                # z step
                torch._foreach_sub_(z, grad, alpha=lr)
            else:
                if 'polyak' in group['mode']:
                    # proj_gap with RAW grad, grad_norm with NORMALIZED grad.
                    # Formula: a_t = [f(y)-f* + <g_raw,z-y>]+ / max(M, ||g~||^2)
                    grad_norm = torch.tensor(0.0)
                    proj_gap = torch.tensor(0.0)
                    for p in active_p:
                        state = self.state[p]
                        z_p = state['z']
                        exp_avg_sq_p = state['exp_avg_sq']
                        g = p.grad

                        exp_avg_sq_p.mul_(beta2).addcmul_(g, g, value=1-beta2)
                        d = exp_avg_sq_p.div(bias_correction2).sqrt_().add_(eps)

                        # proj_gap uses RAW gradient before normalization
                        proj_gap = proj_gap.to(g.device) + torch.sum(g * (z_p - p))

                        # grad_norm = ||D^{-1/2}g||^2 = sum(g^2/D) = sum(g_raw * (g_raw/D))
                        # Must be computed BEFORE in-place normalization.
                        # Matches SCHEDULETA: _norm += torch.sum(grad.mul(grad.div(Dk)))
                        grad_norm = grad_norm.to(g.device) + torch.sum(g * g.div(d))

                        g.div_(d)  # normalize in-place

                        if decay != 0:
                            g.add_(p, alpha=decay)  # WD only for the actual update

                    if not torch.is_tensor(group['M']):
                        M = group['M'] = torch.tensor([group['M']]).to(grad_norm.device)
                    else:
                        M = group['M']

                    if group['polyak_lambda'] is not None and group['polyak_lambda'] > 0:
                        M = group['M'] = group['polyak_lambda'] * group['M'] + (1-group['polyak_lambda']) * grad_norm
                    if M <= 0:
                        M = group['M'] = grad_norm

                    loss_gap = loss - group['polyak_lb']
                    numerator = torch.clamp(loss_gap + proj_gap, min=0)
                    denominator = torch.max(grad_norm, M).item()
                    lr = (numerator / denominator).item() if numerator > 0 else 0.0

                    group['lr'] = lr
                    ckp1 = 1/(k+1)

                    if 'polyak_events' not in self.state:
                        self.state['polyak_events'] = []

                    self.state['polyak_events'].append({
                            'step': group['k'],
                            'grad_norm': grad_norm.item(),
                            'M': M.item(),
                            'used_grad_norm': int((grad_norm > M).item()),
                            'lr': group['lr']
                    })

                    # Second pass: apply updates (grad already normalized in-place above)
                    for p in active_p:
                        z_p = self.state[p]['z']
                        g = p.grad  # already normalized
                        p.lerp_(end=z_p, weight=ckp1)
                        p.add_(g, alpha=lr*(beta1*(1-ckp1)-1))
                        z_p.sub_(g, alpha=lr)
                else:
                    for p in active_p:
                        y = p # Notation to match theory
                        grad = p.grad

                        state = self.state[p]

                        z = state['z']
                        exp_avg_sq = state['exp_avg_sq']

                        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1-beta2)
                        denom = exp_avg_sq.div(bias_correction2).sqrt_().add_(eps)

                        # Reuse grad buffer for memory efficiency
                        grad_normalized = grad.div_(denom)

                        # Weight decay calculated at y
                        if decay != 0:
                            grad_normalized.add_(y, alpha=decay)

                        # These operations update y in-place,
                        # without computing x explicitly.
                        y.lerp_(end=z, weight=ckp1)
                        y.add_(grad_normalized, alpha=lr*(beta1*(1-ckp1)-1))

                        # z step
                        z.sub_(grad_normalized, alpha=lr)

            group['k'] = k+1
        return loss

####. 
class SCHEDULETA(torch.optim.Optimizer):
    def __init__(self,
                 params: 0.0,
                 lr: float=1.0,
                 lmbda: Union[float,None]=9.0,
                 beta2:float=0.999, 
                 eps:float=1e-8,
                 weight_decay: float=0.0,
                 lb: float=0.0,
                 ) -> None:
        """
        ScheduletAdam: Schedule free Adam with a teacher
        Parameters
        ----------
        params : Params
            Model parameters.
        lr : float, optional
            Learning rate cap, by default 1.0.
        lmbda : float or None, optional
            lambda_t from paper, by default 9.0. If set to None, use lambda_t=t
        weight_decay : float, optional
            Weight decay parameter, by default 0.0.
        lb : float, optional
            Lower bound for loss. Zero is often a good guess.
            By default 0.0.
        """
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= beta2 < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if lmbda is not None:
            if lmbda < 0.0:
                raise ValueError("Invalid negative lambda value: {}".format(lmbda))
            self._theoretical_lmbda = False
        else:
            self._theoretical_lmbda = True
        if weight_decay < 0.0:
            raise ValueError("Invalid weight decay: {}".format(weight_decay))
        defaults = dict(lr=lr,
                        lmbda=lmbda,
                        beta2=beta2,
                        eps=eps,
                        weight_decay=weight_decay,
                        weight_sum=0.0
        )
        super(SCHEDULETA, self).__init__(params, defaults)
        self.lb = lb
        # Initialization
        self._number_steps = 0
        self.state['step_size_list'] = list() # for storing the adaptive step size term
        return
    def step(self, closure =None, loss: torch.Tensor=None, lb: float=None):
        """
        Performs a single optimization step.
        Parameters
        ----------
        closure : LossClosure, optional
            A callable that evaluates the model (possibly with backprop) and returns the loss, by default None.
        loss : torch.tensor, optional
            The loss tensor. Use this when the backward step has already been performed. By default None.
        lb : float, optional
            The optimal value for this batch of data. If None, the use the general lower bound from initialization.
        Returns
        -------
        (Stochastic) Loss function value.
        """
        assert (closure is not None) or (loss is not None), "Either loss tensor or closure must be passed."
        assert (closure is None) or (loss is None), "Pass either the loss tensor or the closure, not both."
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        if len(self.param_groups) > 1:
            warnings.warn("More than one param group. step_size_list contains adaptive term of last group.")
            warnings.warn("More than one param group. This might cause issues for the step method.")
        _norm = 0.
        _dot = 0.
        self._number_steps += 1
        average_precon =0
        ############################################################
        for group in self.param_groups:
            lr = group['lr']
            eps = group['eps']
            beta2 = group['beta2']
            for p in group['params']:
                grad = p.grad.data.detach()
                state = self.state[p]

                 # Adam State initialization
                if 'step' not in state:
                    state['step'] = 0
                    # Exponential moving average of squared gradient values
                    state['grad_avg_sq'] = torch.zeros_like(p.data, memory_format=torch.preserve_format).detach()
                    # Initialize Averaging Variables
                    state['z'] = p.detach().clone().to(p.device)
                self._number_steps +=1
                state['step'] += 1 
                grad_avg_sq =  state['grad_avg_sq']

                # Adam EMA updates
                grad_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1-beta2) # = v_k
                # grad_dot_w.mul_(beta1).add_(torch.sum(torch.mul(p.data, grad)), alpha=1-beta1)
                bias_correction2 = 1 - beta2 ** self._number_steps
                Dk = grad_avg_sq.div(bias_correction2).sqrt().add(eps) # = D_k
                z = state['z']
                _dot += torch.sum(torch.mul(grad, z-p.data))
                _norm += torch.sum(grad.mul(grad.div(Dk)))

        num_params = sum(len(group['params']) for group in self.param_groups)
        #################
        # Update
        for group in self.param_groups:
            lr = group['lr']
            lmbda = group['lmbda']
            weight_decay = group['weight_decay']
            beta2 = group['beta2']
            bias_correction2 = 1 - beta2 ** self._number_steps

            # compute lmbda_t
            if self._theoretical_lmbda:
                lmbda = self._number_steps +1     # lmbda_t = t
            ### Compute adaptive step size
            this_lb = self.lb if not lb else lb
            t1 = loss.item() - this_lb + _dot
            eta = max(t1, 0) / _norm
            eta = eta.item() # make scalar
            tau = min(lr, eta)
            ### Update params
            for p in group['params']:
                grad = p.grad.data.detach()
                state = self.state[p]
                grad_avg_sq =  state['grad_avg_sq']
                Dk = grad_avg_sq.div(bias_correction2).sqrt().add(eps)
                # average_precon += (torch.mean(1/Dk)/num_params).item()
                z = state['z']
                if weight_decay > 0.0:
                    z.add_(p.data, alpha= (-lr*weight_decay))  # z = z - lr*wd*x
                # z Update
                z.add_(grad.div(Dk), alpha=-tau)
                # x Update
                p.data.mul_(lmbda/(1+lmbda)).add_(other=z, alpha=1/(1+lmbda))
        ############################################################
        self.state['step_size_list'].append(tau/(1+lmbda))
        return loss