import os, sys
import numpy as np 
import csv
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import pandas as pd 
import pickle
from pathlib import Path
import glob 
import argparse
from data_loaders import DATASETS

def plot_group_RE_params(methods_avg_error, params, d, partition, method, name, result_dir='ablation_images'):
    Path(result_dir).mkdir(parents=True, exist_ok=True)
    markers = ["o", "v", "P", "s", "*"]

    fig = plt.figure(figsize = (10,10), dpi = 200)
    matplotlib.rcParams.update({'font.size': 18})

    ax = fig.add_subplot(111)

    for idx, method in enumerate(methods_avg_error.keys()): 
        avg_error = methods_avg_error[method]

        avg_error_vals_A = avg_error[f'A{partition}']
        avg_error_vals_B = avg_error[f'B{partition}']
        if method == 'RFPCA':
            markersize = 16
        else:
            markersize = 14
        ax.plot(params, avg_error_vals_A, marker=markers[idx], markersize=markersize, label=f"{method} - Group0")
        ax.plot(params, avg_error_vals_B, marker=markers[idx], markersize=markersize, label=f"{method} - Group1")

    ax.set_xlabel(check_param.lower())
    ax.set_xticks(K)
    ylabel = 'Average Subgroup Reconstruction Error'
    ax.set_ylabel(ylabel=ylabel)
    ax.legend(fontsize=20)
    partition_note = 'In-sample' if partition == 'train' else 'Out-of-sample'
    ax.set_title(f'Average reconstruction error\non {DATASETS[dataset]["name"]} ({d}-dim) - {partition_note}')
    filename = f'{result_dir}/{name}_mutiple_k_{partition}.png'
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    print(f'saved to {filename}')
    plt.close(fig)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_names', type=str, nargs='*')
    parser.add_argument('--res_dir', type=str)
    parser.add_argument('--img_dir', type=str)
    parser.add_argument('--prefix', type=str, default=None)
    args = parser.parse_args()

    datasets = args.dataset_names
    prefix = args.prefix
    exp_results_folder = args.res_dir
    img_dir = args.img_dir
    train_percent = 0.5
    check_param = 'K'
    seeds = np.arange(5)

    methods_name = ['PCA', 'FairPCA', 'RFPCA']

    for dataset in datasets:
        methods_avg_errors = {}
        for method in methods_name:
            method_sims = []
            for seed in seeds:
                run_id = f"{method}_results_{dataset}_trainperc{train_percent}_seed{seed}*"
                filepath = glob.glob(f'{exp_results_folder}/{dataset}/{run_id}.pkl')[0]
                with open(filepath, 'rb') as f:
                    method_sim = pickle.load(f)
                    method_sims.append(method_sim)
            assert len(method_sims) == len(seeds)
            criteria = method_sims[0]['test'].keys()
            param_grid = method_sims[0]['params']
            K = [p['K'] for p in param_grid]
            d = param_grid[0]['d']

            avg_errors = {}
            for criterion in criteria:
                avg_errors[criterion] = np.mean([method_sims[i]['test'][criterion] for i in range(len(method_sims))], axis=0)
            methods_avg_errors[method] = avg_errors

        for partition in ['train', 'test']:
            plot_group_RE_params(methods_avg_errors, K, d, 
                            partition, method, name=f'{dataset}_{check_param}', 
                            result_dir=f"{exp_results_folder}/{img_dir}/{dataset}")
