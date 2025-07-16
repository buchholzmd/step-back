"""
Script for generating plots.
"""
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import argparse
import os

from stepback.record import Record
from stepback.utils import get_output_filenames
from stepback.plotting import plot_stability, plot_step_sizes

################# Main setup ###############################
parser = argparse.ArgumentParser(description='Generate step-back plots.')
parser.add_argument('-i', '--id', nargs='?', type=str, default='test', help="The id of the config (its file name).")
args = parser.parse_args()

FIGSIZE = (10,6)
# FIGSIZE = (7,4)

LEGEND_LOC = 'lower left'
LEGEND_OUTSIDE = False

try:
    exp_id = args.id
    save = True
except:
    exp_id = 'cifar100_resnet110'
    save = False

output_names = get_output_filenames(exp_id)
############################################################

#%%
#%matplotlib qt5

plt.rcParams["font.family"] = "serif"
plt.rcParams['font.size'] = 13
plt.rcParams['axes.linewidth'] = 1
plt.rc('text', usetex=True)

#%%
R = Record(output_names)
R.filter(keep={'lr_schedule': 'constant', 'lr_schedule': 'wsd'})

# R.filter(drop={'name': ['momo-adam-star', 'momo-star']})
# R.filter(drop={'name': ['adabelief', 'adabound', 'lion', 'prox-sps']}) 
# R.filter(keep={'lr_schedule': 'constant'}) 


R = Record(output_names)
R.filter(keep={'lr_schedule': 'constant', 'lr_schedule': 'wsd'})

# R.filter(drop={'name': ['momo-adam-star', 'momo-star']})
# R.filter(drop={'name': ['adabelief', 'adabound', 'lion', 'prox-sps']}) 
# R.filter(keep={'lr_schedule': 'constant'})

# R.filter(drop={'name': ['adamw', 'schedule-free-adam', 'schedulet-adam']})
# R.filter(drop={'name': ['sgd-m', 'schedule-free', 'schedulet']})

keep_list = {
    'sgd':  {'name': ['sgd-m', 'schedule-free', 'schedulet']},
    'adam': {'name': ['adamw', 'schedule-free-adam', 'schedulet-adam']}
}

for k in keep_list.keys():
    R_opt = R.copy()

    R_opt.filter(keep=keep_list[k])

    base_df = R_opt.base_df                                 # base dataframe for all plots
    id_df = R_opt.id_df                                     # dataframe with the optimizer setups that were run

    # _ = R.plot_metric(s='val_score', log_scale=False, legend=True)

    #%% plot training curves for a subset of runs:

    # takes 3 best runs per methods
    best = base_df[base_df.epoch==base_df.epoch.max()].groupby('name')['val_score'].nlargest(3)
    #best = base_df[base_df.epoch==base_df.epoch.max()].groupby('name')['train_loss'].nsmallest(3)
    ixx = base_df.id[best.index.levels[1]]
    df1 = base_df.loc[base_df.id.isin(ixx),:]

    # y0 = 0.3 if 'cifar100_resnet110' in exp_id else 0.4 if 'cifar10_vit' in exp_id else 0.75
    y0 = 0.9 * df1['val_score'].min()

    fig, ax = R_opt.plot_metric(df=df1, 
                                s='val_score', 
                                ylim=(y0, 1.05*df1.val_score.max()), 
                                log_scale=False, 
                                figsize=FIGSIZE, 
                                legend=False,
                                legend_loc=LEGEND_LOC,
                                legend_outside=LEGEND_OUTSIDE,
                                )
    fig.subplots_adjust(top=0.975,bottom=0.16,left=0.16,right=0.975)

    os.makedirs('output/plots/' + exp_id, exist_ok=True)

    if save:
        fig.savefig('output/plots/' + exp_id + f'/all_val_score_{k}.pdf')

    fig, ax = R_opt.plot_metric(df=df1, 
                                s='train_loss', 
                                log_scale=True, 
                                figsize=FIGSIZE, 
                                legend=False,
                                legend_loc=LEGEND_LOC,
                                legend_outside=LEGEND_OUTSIDE,
                                )
    fig.subplots_adjust(top=0.975,bottom=0.16,left=0.17,right=0.975)
    if save:
        fig.savefig('output/plots/' + exp_id + f'/all_train_loss_{k}.pdf')


#%% stability plots

fig, axs = plot_stability(R, 
                          score='val_score', 
                          xaxis='lr', sigma=1, 
                          legend=None, 
                          cutoff=None, 
                          figsize=FIGSIZE, 
                          save=save,
                          legend_loc=LEGEND_LOC,
                          legend_outside=LEGEND_OUTSIDE
                        )
fig, axs = plot_stability(R, 
                          score='train_loss', 
                          xaxis='lr', 
                          sigma=1, 
                          legend=None, 
                          cutoff=None, 
                          figsize=FIGSIZE, 
                          save=save,
                          legend_loc=LEGEND_LOC,
                          legend_outside=LEGEND_OUTSIDE,
                        )
fig, axs = plot_stability(R, 
                          score=['train_loss', 'val_score'], 
                          xaxis='lr', 
                          sigma=1, 
                          legend=None, 
                          cutoff=None, 
                          figsize=(FIGSIZE[0],2*FIGSIZE[1]), 
                          save=save,
                          legend_loc=LEGEND_LOC,
                          legend_outside=LEGEND_OUTSIDE,
                        )

#%% plots the adaptive step size
### THIS PLOT IS ONLY RELEVANT FOR METHODS WITH ADAPTIVE STEP SIZE
###################################

# if 'cifar10_resnet20' in exp_id:
#     _ = plot_step_sizes(R, method='momo', grid=(3,3), start=None, stop=None, save=save)
#     _ = plot_step_sizes(R, method='momo-adam', grid=(3,2), start=1, stop=None, save=save)
# elif 'cifar10_vgg16' in exp_id:
#     _ = plot_step_sizes(R, method='momo', grid=(3,3), start=2, stop=11, save=save)
#     _ = plot_step_sizes(R, method='momo-adam', grid=(3,3), start=2, stop=11, save=save)
# elif 'mnist_mlp' in exp_id:
#     _ = plot_step_sizes(R, method='momo', grid=(3,2), start=1, stop=None, save=save)
#     _ = plot_step_sizes(R, method='momo-adam', grid=(3,2), start=None, stop=None, save=save)
# elif 'cifar100_resnet110' in exp_id:
#     _ = plot_step_sizes(R, method='momo', grid=(3,2), start=1, stop=7, save=save)
#     _ = plot_step_sizes(R, method='momo-adam', grid=(3,2), start=1, stop=7, save=save)
# elif 'cifar10_vit' in exp_id:
#     _ = plot_step_sizes(R, method='momo', grid=(2,2), start=1, stop=5, save=save)
#     _ = plot_step_sizes(R, method='momo-adam', grid=(2,2), start=None, stop=None, save=save)


# %%
