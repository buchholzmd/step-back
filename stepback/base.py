import os
import tqdm
import numpy as np
import torch
import copy
import time
import datetime
import warnings
from typing import Union
import wandb

from torch.utils.data import DataLoader

from .datasets.main import get_dataset, infer_shapes
from .models.main import get_model
from .optim.main import get_optimizer, get_scheduler, optimal_lr_sched
from .metrics import Loss

from .utils import l2_norm, grad_norm, ridge_opt_value, logreg_opt_value
from .defaults import DEFAULTS

class Base:
    def __init__(self, name: str, 
                 config: dict, 
                 device: str=DEFAULTS.device, 
                 data_dir: str=DEFAULTS.data_dir,
                 num_workers: int=DEFAULTS.num_workers,
                 data_parallel: Union[list, None]=DEFAULTS.data_parallel,
                 verbose: bool=DEFAULTS.verbose,
                 wandb: bool=DEFAULTS.wandb,
                 project: str='schedule-free'):
        """The main class. Performs one single training run plus evaluation.

        Parameters
        ----------
        name : str
            A name for this run. Currently not used later.
        config : dict
            A config containing dataset, model, optimizer information.
            Needs to have the keys ['loss_func', 'score_func', 'opt'].
        device : str, optional
            Device string, by default 'cuda'
            If 'cuda' is specified, but not available on system, it switches to CPU.
        data_dir : str, optional
            Directory where datasets can be found, by default 'data/'
        num_workers : int, optional
            Number of workers for DataLoader, by default 0
        data_parallel : Union[list, None], optional
            If not None, this specifies the device ids for DataParallel mode in Pytorch. By default None.
            See https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html.
        verbose : bool, optional
            Verbose mode flag, by default False.
            If True, prints progress bars, model architecture and other useful information.
        wandb : str, optional
            If not None, this specifies the entity for wandb logging. By default None, which means no wandb logging.
            If specified, enables wandb logging for real-time visualization of metrics during training and validation.
        project : str, optional
            wandb project name for grouping experiments, by default 'schedule-free'.
        """
        
        self.name = name
        self.config = copy.deepcopy(config)
        self.data_dir = data_dir
        self.num_workers = num_workers
        self.data_parallel = data_parallel
        self.verbose = verbose
        self.wandb = wandb
        self.project = project
        self.global_step = 0
        
        model_dir = f"output/{self.config['dataset']}_{self.config['model']}"

        opt_dir = f"{self.config['opt']['name']}_{self.config['opt']['lr']}_{self.config['opt']['lr_schedule']}"
        if 'warmup' in self.config['opt'].keys():
            opt_dir += f"_{self.config['opt']['warmup']}"
        if 'cooldown' in self.config['opt'].keys():
            opt_dir += f"_{self.config['opt']['cooldown']}"
        if 'transition' in self.config['opt'].keys():
            opt_dir += f"_{self.config['opt']['transition']}"
        if 'slope' in self.config['opt'].keys():
            opt_dir += f"_{self.config['opt']['slope']}"

        self.output_dir = os.path.join(model_dir, opt_dir)
        os.makedirs(self.output_dir, exist_ok=True)

        print("CUDA available? ", torch.cuda.is_available())
        
        if torch.cuda.is_available():
            self.device = torch.device(device)
            self.pin_memory = True
        else:
            self.device = torch.device('cpu')
            self.pin_memory = False
            
        print("Using device: ", self.device)
        
        self.seed = 1234567
        self.run_seed = 456789 + config.get('run_id', 0)
        torch.backends.cudnn.benchmark = False
        # see https://pytorch.org/docs/stable/notes/randomness.html#reproducibility
        
        self.check_config()

        # Create ditionary for results
        self.results = {'config': self.config,
                        'history': {},
                        'summary': {}}
        
        self.results['summary']['num_workers'] = self.num_workers
        self.results['summary']['data_parallel'] = 'true' if self.data_parallel else 'false'
        
        return
        
    def check_config(self):
        
        # create defaults for missing config keys
        if 'batch_size' not in self.config.keys():
            self.config['batch_size'] = 1
        
        if 'dataset_kwargs' not in self.config.keys():
            self.config['dataset_kwargs'] = dict()

        if 'model_kwargs' not in self.config.keys():
            self.config['model_kwargs'] = dict()
        
        # check necessary config keys
        for k in ['loss_func', 'score_func', 'opt']:
            assert k in self.config.keys(), f"You need to specify {k} in the config file."

        return
    
    def _setup_data(self):
        """Loads training and validation set. Creates DataLoader for training."""
        self.train_set = get_dataset(config=self.config, split='train', seed=self.seed, path=self.data_dir)
        self.val_set = get_dataset(config=self.config, split='val', seed=self.run_seed, path=self.data_dir)
        
        self.results['summary']['input_dim'], self.results['summary']['output_dim'] = infer_shapes(self.train_set)
        
        # construct train loader
        _gen = torch.Generator()
        _gen.manual_seed(self.run_seed)
        self.train_loader = DataLoader(self.train_set, drop_last=True, shuffle=True, generator=_gen,
                                       batch_size=self.config['batch_size'],
                                       num_workers=self.num_workers,
                                       pin_memory=self.pin_memory
        )
        
        return

    def _setup_model(self):
        """Initializes the model."""
        torch.manual_seed(self.seed) # Reseed to have same initialization   
        torch.cuda.manual_seed_all(self.seed)        
        
        self.model = get_model(config=self.config, 
                               input_dim=self.results['summary'].get('input_dim',[]), 
                               output_dim=self.results['summary'].get('output_dim',[])
        )
        
        self.model.to(self.device)

        if self.data_parallel is not None:
            devices = [int(d) for d in self.data_parallel]
            self.model = torch.nn.DataParallel(self.model, device_ids=devices)

        self.init_params = {name: p.detach().cpu().numpy().flatten() for name, p in self.model.named_parameters()}
        self.best_params = {name: p.detach().cpu().numpy().flatten() for name, p in self.model.named_parameters()}
        np.savez(f'{self.output_dir}/init_model_params.npz', **self.init_params)

        self.grad_norm_history = []

        return
        

    def setup(self):

        #============ Data =================
        self._setup_data()
               
        #============ Model ================
        self._setup_model()
        
        if self.verbose:
            print(self.model)

        #============ Loss function ========
        self.training_loss = Loss(name=self.config['loss_func'], backwards=True)
        self.init_loss = None
        self.best_loss = np.inf
        
        #============ Optimizer ============
        self.config['opt']['total_steps'] = self.config['max_epoch'] * len(self.train_loader)
        opt_obj, hyperp = get_optimizer(self.config['opt'])
        
        self._init_opt(opt_obj, hyperp)
        
        self.lr_sched = self.config['opt'].get('lr_schedule')
        self.sched = get_scheduler(self.config['opt'], self.opt)

        if self.config['opt'].get('lr_schedule') == 'optimal':
            self.optimal_lr_sched = optimal_lr_sched
        else:
            self.optimal_lr_sched = None
        
        #============ Results ==============
        opt_val = self._compute_opt_value()
        
        # Store useful information as summary
        if opt_val is not None:
            self.results['summary']['opt_val'] = opt_val
        
        #============ WandB Logging ========
        if self.wandb is not None:
            wandb.init(
                entity=self.wandb,
                project=self.project,
                name=f"{self.config['dataset']}_{self.config['model']}_{self.config['opt']['name']}_lr{self.config['opt']['lr']}",
                config={
                    'dataset': self.config['dataset'],
                    'model': self.config['model'],
                    'optimizer': self.config['opt']['name'],
                    **self.config['opt']
                }
            )

        return 
    
    def _init_opt(self, opt_obj, hyperp, opt_name=None):
        """Initializes the opt object. If your optimizer needs custom commands, add them here."""
        
        self.opt = opt_obj(params=self.model.parameters(), **hyperp)

        self.lr_history = [self.config['opt']['lr']]
        
        print(self.opt)        
        return
    
    def run(self):
        start_time = str(datetime.datetime.now())
        score_list, batch_list = [], [{'learning_rate': self.config['opt']['lr']}]
        self._epochs_trained = 0
        
        for epoch in range(self.config['max_epoch']):
            if self.optimal_lr_sched is not None:
                print(f"Epoch {epoch}, current learning rate", batch_list[-1]['learning_rate'])
            else:
                print(f"Epoch {epoch}, current learning rate", self.sched.get_last_lr()[0])

            # Record metrics
            score_dict = {'epoch': epoch}
            if self.optimal_lr_sched is not None:
                score_dict['learning_rate'] = batch_list[-1]['learning_rate']
            else:
                score_dict['learning_rate'] = self.sched.get_last_lr()[0] # must be stored before sched.step()
                           
            # Train one epoch
            s_time = time.time()
            epoch_batch_list = self.train_epoch()
            e_time = time.time()
            
            # Record metrics
            score_dict['train_epoch_time'] = e_time - s_time       
            score_dict['model_norm'] = l2_norm(self.model)        
            score_dict['grad_norm'] = grad_norm(self.model)
                
            # Validation
            with torch.no_grad():
                
                metric_dict = {'loss': Loss(self.config['loss_func'], backwards=False), 
                               'score': Loss(self.config['score_func'], backwards=False)}      
                
                train_dict = self.evaluate(self.train_set, 
                                           metric_dict = metric_dict,
                                           )
            
                val_dict = self.evaluate(self.val_set, 
                                         metric_dict = metric_dict,
                                         )             
                       
                # Record metrics
                score_dict.update(train_dict)
                score_dict.update(val_dict)
                
                # Record metrics specific to MoMo methods 
                if self.opt.state.get('step_size_list'):
                    score_dict['step_size_list'] = [float(np.format_float_scientific(t,5)) for t in self.opt.state['step_size_list']] 
                    self.opt.state['step_size_list'] = list()
                # fstar estimator (could be zero)
                if self.opt.state.get('fstar', None) is not None:
                    score_dict['fstar'] = self.opt.state['fstar']
                if self.opt.state.get('h', None) is not None:
                    score_dict['h'] = self.opt.state['h']

                # Add score_dict to score_list
                score_list += [score_dict]
                batch_list += epoch_batch_list
                
                # Log epoch metrics to wandb (organized by section)
                if self.wandb is not None:
                    # Extract train metrics
                    train_metrics = {f'train/epoch/{k}': v for k, v in score_dict.items() 
                                    if k.startswith('train_')}
                    # Extract val metrics
                    val_metrics = {f'val/{k}': v for k, v in score_dict.items() 
                                  if k.startswith('val_')}
                    # Extract general metrics (epoch, learning_rate, model_norm, grad_norm, etc.)
                    general_metrics = {f'train/epoch/{k}': v for k, v in score_dict.items() 
                                      if not k.startswith(('train_', 'val_'))}
                    
                    wandb.log({**train_metrics, **val_metrics, **general_metrics}, step=self.global_step)
                    
                    # Log optimizer state if available (polyak_events is a list, take the last one)
                    if self.opt.state.get('polyak_events') and len(self.opt.state['polyak_events']) > 0:
                        last_event = self.opt.state['polyak_events'][-1]  # Most recent polyak event
                        opt_metrics = {f'optimizer/polyak/{k}': v for k, v in last_event.items()}
                        wandb.log(opt_metrics, step=self.global_step)
                        # Clear for next epoch
                        self.opt.state['polyak_events'] = []
            
            self._epochs_trained += 1
        
        end_time = str(datetime.datetime.now())
        
        # ==== store =====================
        self.results['history'] = copy.deepcopy(score_list)
        self.results['batch_history'] = copy.deepcopy(batch_list)
        self.results['summary']['start_time'] = start_time
        self.results['summary']['end_time'] = end_time
        
        # ==== finalize wandb ============
        if self.wandb is not None:
            wandb.finish()
        
        return

    def train_epoch(self):
        """
        Train one epoch.
        """
                
        self.model.train()

        if 'schedulefree' in type(self.opt).__name__.lower():
            self.opt.train()

        pbar = tqdm.tqdm(self.train_loader, disable=(not self.verbose))
        
        timings_model = list()
        timings_dataloader = list()

        batch_list = []

        t0 = time.time()

        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            data_t0 = time.time()
            data, targets = batch['data'].to(device=self.device), batch['targets'].to(device=self.device)
            
            # Forward and Backward
            t1 = time.time()                                        # model timing starts
            self.opt.zero_grad()    
            out = self.model(data).to(device=self.device)

            if len(out.shape) <= 1:
                warnings.warn(f"Shape of model output is {out.shape}, recommended to have shape [batch_size, ..].")
            
            # see optim/README.md for explanation 
            if hasattr(self.opt,"prestep"):
                ind = batch['ind'].to(device=self.device)           # indices of batch members
                self.opt.prestep(out, targets, ind, self.training_loss.name)
            
            if self.device != torch.device('cpu'):
                torch.cuda.synchronize()
            timings_dataloader.append(t1-t0) 
            t0 = time.time()                                        # model timing ends
            timings_model.append(t0-t1)

            # Record metrics
            global_idx = batch_idx + self._epochs_trained * len(self.train_loader)
            batch_dict = {'iteration': global_idx}
            
            # Update learning rate
            if self.optimal_lr_sched is not None:
                if self.training_loss._flatten_out:
                    out = out.view(-1)
                
                if self.training_loss._flatten_target:
                    targets = targets.view(-1)
                    
                loss = self.training_loss.criterion(out, targets)
                loss_val = loss.item()

                if loss.requires_grad:
                    loss.backward()

                # Use custom LR function
                current_grad_norm = grad_norm(self.model)
                if self.grad_norm_history == []:
                    self.grad_norm_history = [current_grad_norm]

                if self.init_loss is None:
                    self.init_loss = loss_val

                if loss_val < self.best_loss:
                    self.best_loss = loss_val
                    
                    self.best_params = {name: p.detach().cpu().numpy().flatten() for name, p in self.model.named_parameters()}
                    np.savez(f'{self.output_dir}/best_model_params.npz', **self.best_params)

                init_params = torch.cat([torch.tensor(self.init_params[k].ravel()) for k in self.init_params])
                best_params = torch.cat([torch.tensor(self.best_params[k].ravel()) for k in self.best_params])
                
                new_lr = self.optimal_lr_sched(self.lr_history,
                                               self.grad_norm_history,
                                               current_grad_norm,
                                               self.init_loss,
                                               self.best_loss, 
                                               init_params,
                                               best_params,
                                               use_c=False
                )
                for param_group in self.opt.param_groups:
                    param_group['lr'] = new_lr

                self.lr_history += [new_lr.item()]
                self.grad_norm_history += [current_grad_norm]

                # Here the magic happens
                self.opt.step()

                batch_dict['learning_rate'] =  new_lr.item()
            else:
                # For polyak/SPS optimizers, use unreduced (sum) loss for correct batch scaling
                opt_name = type(self.opt).__name__.lower()
                if 'polyak' in opt_name or 'sps' in opt_name:
                    # unreduced_loss = Loss(self.config['loss_func'], backwards=True, reduction='none')
                    unreduced_loss = Loss(self.config['loss_func'], backwards=True, reduction='mean')
                    closure = lambda: unreduced_loss.compute(out, targets)
                else:
                    closure = lambda: self.training_loss.compute(out, targets)

                # Here the magic happens
                loss_val = self.opt.step(closure=closure)

                current_grad_norm = grad_norm(self.model)

                if self.init_loss is None:
                    self.init_loss = loss_val

                if loss_val < self.best_loss:
                    self.best_loss = loss_val
                    
                    self.best_params = {name: p.detach().cpu().numpy().flatten() for name, p in self.model.named_parameters()}
                    np.savez(f'{self.output_dir}/best_model_params.npz', **self.best_params)

                # Use standard scheduler
                self.sched.step()
                batch_dict['learning_rate'] = self.sched.get_last_lr()[0] # must be stored before sched.step()
              
            pbar.set_description(f'Training - loss={loss_val:.3f} - time data: last={timings_dataloader[-1]:.3f},(mean={np.mean(timings_dataloader):.3f}) - time model+step: last={timings_model[-1]:.3f}(mean={np.mean(timings_model):.3f})')
            
            # Record metrics   
            batch_dict['model_norm'] = l2_norm(self.model) 
            batch_dict['grad_norm']  = current_grad_norm
            batch_dict['train_loss'] = loss_val.item() if hasattr(loss_val, 'item') else loss_val

            batch_list += [batch_dict]
            
            # Log per-iteration batch metrics to wandb
            if self.wandb is not None:
                batch_metrics = {f'train/iteration/{k}': v for k, v in batch_dict.items()}
                wandb.log(batch_metrics, step=self.global_step)
                self.global_step += 1

        return batch_list
    
    def evaluate(self, dataset, metric_dict):
        """
        Evaluate model for a given dataset (train or val), and for several metrics.
        
        metric_dict:
            Should have the form {'metric_name1': metric1, 'metric_name2': metric2, ...}
        """
        
        dl = torch.utils.data.DataLoader(dataset, drop_last=False, 
                                         batch_size=self.config['batch_size'],
                                         num_workers=self.num_workers,
                                         pin_memory=self.pin_memory
                                         )
        pbar = tqdm.tqdm(dl, disable=(not self.verbose))
        
        self.model.eval()

        if 'schedulefree' in type(self.opt).__name__.lower():
            self.opt.eval()

        score_dict = dict(zip(metric_dict.keys(), np.zeros(len(metric_dict))))

        timings_model = list()
        timings_dataloader = list()
             
        t0 = time.time()
       
        for batch in pbar:
            # get batch and compute model output
            data, targets = batch['data'].to(device=self.device), batch['targets'].to(device=self.device)

            t1 = time.time()
            out = self.model(data)
            
            for _met, _met_fun in metric_dict.items():
                # metric takes average over batch ==> multiply with batch size
                score_dict[_met] += _met_fun.compute(out, targets).item() * data.shape[0] 
            
            timings_dataloader.append(t1-t0)                                         
            if self.device != torch.device('cpu'):
                torch.cuda.synchronize()        
            t0 = time.time()
            timings_model.append(t0-t1)    

            pbar.set_description(f'Validating {dataset.split}')
            pbar.set_description(f'Validating {dataset.split} - time data: last={timings_dataloader[-1]:.3f}(mean={np.mean(timings_dataloader):.3f}) - time model: last={timings_model[-1]:.3f}(mean={np.mean(timings_model):.3f})')

        for _met in metric_dict.keys():
            # Get from sum to average
            score_dict[_met] = float(score_dict[_met] / len(dl.dataset))
            
            # add split in front of names
            score_dict[dataset.split + '_' + _met] = score_dict.pop(_met)
        
        return score_dict

    def save_checkpoint(self, path):
        """See https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html"""
        torch.save({'epoch': self._epochs_trained,
                    'model_state_dict': self.model.state_dict(),
                    'opt_state_dict': self.opt.state_dict(),
                    }, 
                   path + self.name + '.mt')

        return         

    def _compute_opt_value(self):
        """
        For linear model, the problem is convex and we can compute the optimal value
        """
        if self.config['model'] == 'linear':
            
            fit_intercept = (self.model[0].bias is not None)

            if fit_intercept and self.config['opt'].get('weight_decay', 0) > 0:
                warnings.warn("Using bias and weight decay. Note that the implementation her will also penalize the bias.")
            
            if self.config['loss_func'] == 'squared':
                opt_val = ridge_opt_value(X=self.train_set.dataset.tensors[0].detach().numpy(),
                                          y=self.train_set.dataset.tensors[1].detach().numpy(),
                                          lmbda = self.config['opt'].get('weight_decay', 0),
                                          fit_intercept = fit_intercept
                                          )
            elif self.config['loss_func'] == 'logistic':
                opt_val = logreg_opt_value(X=self.train_set.dataset.tensors[0].detach().numpy(),
                                           y=self.train_set.dataset.tensors[1].detach().numpy().astype(int).reshape(-1),
                                           lmbda = self.config['opt'].get('weight_decay', 0),
                                           fit_intercept = fit_intercept
                                           )
            else:
                opt_val = None
        else:
            opt_val = None
            
        return opt_val
    
    

        



