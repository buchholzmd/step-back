import torch
from torch.optim.lr_scheduler import LambdaLR, StepLR
import warnings
from typing import Tuple

from .momo import Momo
from .momo_adam import MomoAdam
from .sps import SPS
from .adabound import AdaBoundW
from .adabelief import AdaBelief
from .lion import Lion
from .schedule_free import SGDScheduleFree, AdamWScheduleFree
from .schedulet import SGDSchedulet, AdamWSchedulet

def get_optimizer(opt_config: dict) -> Tuple[torch.optim.Optimizer, dict]:
    """
    Main function mapping opt configs to an instance of torch.optim.Optimizer and a dict of hyperparameter arguments (lr, weight_decay,..).  
    
    For all hyperparameters which are not specified, we use PyTorch default.
    """
    
    name = opt_config['name']
    
    if opt_config.get('lr') is None:
        warnings.warn("You have not specified a learning rate. A default value of 1e-3 will be used.")
    
    if name == 'sgd':
        opt_obj = torch.optim.SGD
        
        hyperp = {'lr': opt_config.get('lr', 1e-3),
                  'weight_decay': opt_config.get('weight_decay', 0)
                  }
        
    elif name == 'sgd-m':
        opt_obj = torch.optim.SGD
        # sgd-m with exp. weighted average should have dampening = momentum
        if opt_config.get('dampening') == 'momentum':
            dampening = opt_config.get('momentum', 0.9)
        else:
            dampening = opt_config.get('dampening', 0)
            
        hyperp = {'lr': opt_config.get('lr', 1e-3),
                  'weight_decay': opt_config.get('weight_decay', 0),
                  'momentum': opt_config.get('momentum', 0.9),
                  'nesterov': False,
                  'dampening': dampening
                  }

    elif name == 'sgd-nesterov':
        opt_obj = torch.optim.SGD
        hyperp = {'lr': opt_config.get('lr', 1e-3),
                  'weight_decay': opt_config.get('weight_decay', 0),
                  'momentum': opt_config.get('momentum', 0.9),
                  'nesterov': True,
                  'dampening': opt_config.get('dampening', 0)
                  }
               
    elif name == 'adam':
        opt_obj = torch.optim.Adam
        hyperp = {'lr': opt_config.get('lr', 1e-3),
                  'weight_decay': opt_config.get('weight_decay', 0),
                  'betas': opt_config.get('betas', (0.9, 0.999)),
                  'eps': opt_config.get('eps', 1e-8)
                  }
    
    elif name == 'adamw':
        opt_obj = torch.optim.AdamW
        hyperp = {'lr': opt_config.get('lr', 1e-3),
                  'weight_decay': opt_config.get('weight_decay', 0),
                  'betas': opt_config.get('betas', (0.9, 0.999)),
                  'eps': opt_config.get('eps', 1e-8)
                  }
    
    elif name == 'momo':
        opt_obj = Momo
        hyperp = {'lr': opt_config.get('lr', 1e-3),
                  'weight_decay': opt_config.get('weight_decay', 0),
                  'beta': opt_config.get('beta', 0.9),
                  'lb': opt_config.get('lb', 0.),
                  'bias_correction': opt_config.get('bias_correction', False),
                  'use_fstar': False
                  }
    
    elif name == 'momo-adam':
        opt_obj = MomoAdam
        hyperp = {'lr': opt_config.get('lr', 1e-3),
                  'weight_decay': opt_config.get('weight_decay', 0),
                  'betas': opt_config.get('betas', (0.9, 0.999)),
                  'eps': opt_config.get('eps', 1e-8),
                  'lb': opt_config.get('lb', 0.),
                  'divide': opt_config.get('divide', True),
                  'use_fstar': False
                  }
        
    elif name == 'momo-star':
        opt_obj = Momo
        hyperp = {'lr': opt_config.get('lr', 1e-3),
                  'weight_decay': opt_config.get('weight_decay', 0),
                  'beta': opt_config.get('beta', 0.9),
                  'lb': opt_config.get('lb', 0.),
                  'bias_correction': opt_config.get('bias_correction', False),
                  'use_fstar': True
                  }
        
    elif name == 'momo-adam-star':
        opt_obj = MomoAdam
        hyperp = {'lr': opt_config.get('lr', 1e-3),
                  'weight_decay': opt_config.get('weight_decay', 0),
                  'betas': opt_config.get('betas', (0.9, 0.999)),
                  'eps': opt_config.get('eps', 1e-8),
                  'lb': opt_config.get('lb', 0.),
                  'divide': opt_config.get('divide', True),
                  'use_fstar': True
                  }
          
    elif name == 'prox-sps':
        opt_obj = SPS
        hyperp = {'lr': opt_config.get('lr', 1e-3),
                  'weight_decay': opt_config.get('weight_decay', 0),
                  'lb': opt_config.get('lb', 0.),
                  'prox': True
                  }
    
    elif name == 'adabound':
        opt_obj = AdaBoundW
        
        hyperp = {'lr': opt_config.get('lr', 1e-3),
                  'weight_decay': opt_config.get('weight_decay', 0),
                  'betas': opt_config.get('betas', (0.9, 0.999)),
                  'eps': opt_config.get('eps', 1e-8),
                  'final_lr': opt_config.get('final_lr', 0.1)
                  }

    elif name == 'adabelief':
        opt_obj = AdaBelief
        hyperp = {'lr': opt_config.get('lr', 1e-3),
                  'weight_decay': opt_config.get('weight_decay', 0),
                  'betas': opt_config.get('betas', (0.9, 0.999)),
                  'eps': opt_config.get('eps', 1e-16),
                  }
        
    elif name == 'lion':
        opt_obj = Lion
        hyperp = {'lr': opt_config.get('lr', 1e-3),
                  'weight_decay': opt_config.get('weight_decay', 0),
                  'betas': opt_config.get('betas', (0.9, 0.99)),
                  }
        
    elif name == 'schedule-free':
        opt_obj = SGDScheduleFree
        hyperp = {'lr': opt_config.get('lr', 1.0),
                  'momentum': opt_config.get('weight_decay', 0.9),
                  'weight_decay': opt_config.get('weight_decay', 0),
                  'warmup_steps': opt_config.get('warmup_steps', 0),
                  'r': opt_config.get('r', 0),
                  'weight_lr_power': opt_config.get('weight_lr_power', 2.0),
                  }

    elif name == 'schedulet':
        opt_obj = SGDSchedulet
        hyperp = {'lr': opt_config.get('lr', 1.0),
                  'momentum': opt_config.get('weight_decay', 0.9),
                  'weight_decay': opt_config.get('weight_decay', 0),
                  'warmup_steps': opt_config.get('warmup_steps', 0),
                  }

    elif name == 'schedule-free-adam':
        opt_obj = AdamWScheduleFree
        hyperp = {'lr': opt_config.get('lr', 0.0025),
                  'betas': opt_config.get('betas', (0.9, 0.999)),
                  'eps': opt_config.get('eps', 1e-8),
                  'weight_decay': opt_config.get('weight_decay', 0),
                  'warmup_steps': opt_config.get('warmup_steps', 0),
                  'r': opt_config.get('r', 0),
                  'weight_lr_power': opt_config.get('weight_lr_power', 2.0),
                  }

    elif name == 'schedulet-adam':
        opt_obj = AdamWSchedulet
        hyperp = {'lr': opt_config.get('lr', 0.0025),
                  'betas': opt_config.get('betas', (0.9, 0.999)),
                  'eps': opt_config.get('eps', 1e-8),
                  'weight_decay': opt_config.get('weight_decay', 0),
                  'warmup_steps': opt_config.get('warmup_steps', 0),
                  }
    else:
        raise KeyError(f"Unknown optimizer name {name}.")
        
    return opt_obj, hyperp

def get_wsd_lambda(warmup, cooldown, total_epochs):
    assert(0.0 < warmup < 1.0)
    assert(0.0 < cooldown < 1.0)
    assert(warmup + cooldown <= 1.0)

    warmup_epochs = int(warmup * total_epochs)
    decay_epochs = int(cooldown * total_epochs)
    stable_epochs = total_epochs - warmup_epochs - decay_epochs

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # Linear warmup: from 0 to 1
            return (epoch + 1) / warmup_epochs
        elif epoch < warmup_epochs + stable_epochs:
            # Stable: LR = base LR
            return 1.0
        else:
            # Linear decay: from 1 to 0
            decay_epoch = epoch - (warmup_epochs + stable_epochs)
            return max(0.0, 1.0 - decay_epoch / decay_epochs)
    
    return lr_lambda

def get_scheduler(config: dict, opt: torch.optim.Optimizer) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Main function mapping to a learning rate scheduler.
    """
    # if not specified, use constant step sizes
    name = config.get('lr_schedule', 'constant')
    
    if name == 'constant':
        lr_fun = lambda epoch: 1 # this value is multiplied with initial lr
        scheduler = LambdaLR(opt, lr_lambda=lr_fun)
    
    elif name == 'linear':
        lr_fun = lambda epoch: 1/(epoch+1) # this value is multiplied with initial lr
        scheduler = LambdaLR(opt, lr_lambda=lr_fun)
        
    elif name == 'sqrt':
        lr_fun = lambda epoch: (epoch+1)**(-1/2) # this value is multiplied with initial lr
        scheduler = LambdaLR(opt, lr_lambda=lr_fun)
        
    elif 'exponential' in name:
        # use sth like 'exponential_60_0.5': decay by factor 0.5 every 60 epochs
        step_size = int(name.split('_')[1])
        gamma = float(name.split('_')[2])
        scheduler = StepLR(opt, step_size=step_size, gamma=gamma)

    elif 'wsd' in name:
        warmup = config.get('warmup', 0.0)
        cooldown = config.get('cooldown', 0.0)
        total_epochs = config.get('max_epoch', 100)

        lr_fun = get_wsd_lambda(warmup=warmup, cooldown=cooldown, total_epochs=total_epochs)
        scheduler = LambdaLR(opt, lr_lambda=lr_fun)
        
    else:
        raise ValueError(f"Unknown learning rate schedule name {name}.")
    
    return scheduler