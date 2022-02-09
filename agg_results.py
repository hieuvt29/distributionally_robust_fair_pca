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
import json
from data_loaders import DATASETS

def dataset_col_present(methods_name, dataset, methods_avg):
    msg = ''
    cr_vals = {}

    for idx, method in enumerate(methods_name):
        for cr in methods_avg[method].keys():
            if cr not in cr_vals:
                cr_vals[cr] = []

            cr_vals[cr].append(methods_avg[method][cr])

    for idx, method in enumerate(methods_name):
        method_avg = methods_avg[method]
        if idx == 0:
            msg += f"{DATASETS[dataset]['name']} & "

        crs = np.array(sorted([cr for (cr, val) in method_avg.items()]), dtype='object')#[[0, 2, 1]]
        vals = np.array([method_avg[cr] for cr in crs])
        # print(crs)
        msg += " & ".join([f"$\\bf {val:.4f}$" if (val == min(cr_vals[cr])) else f"${val:.4f}$" for cr, val in zip(crs, vals)])
        
        if idx < len(methods_name) - 1:
            msg += " & "
        else:
            msg += " \\\ "
    print(msg)

def dataset_block_present(method, dataset, method_avg, method_std, method_sims, show_err, show_std=False):
    if method == 'RFPCA':
        if show_err:
            if show_std:
                msg = f"{method:18}" + "\t".join([f"{val:.4f}±{std:.4f}" for ((cr, val), (cr, std)) in zip(method_avg.items(), method_std.items())])
            else:
                msg = f"{method:18}" + "\t".join([f"{val:.4f}" for (cr, val) in method_avg.items()])
            print(msg)
        else:
            msg = f"{'RFPCA':18}"
            for i in range(len(method_sims)):
                msg += f"alpha={method_sims[i]['test']['params']['alpha']:.2f}\t"
            msg += '\n'
            msg += f"{'RFPCA':18}"
            for i in range(len(method_sims)):
                msg += f"lambda={method_sims[i]['test']['params']['lamda']:.2f}\t"
            print(msg)
    elif method == 'FairPCA':
        if show_err:
            if show_std:
                msg = f"{method:18}" + "\t".join([f"{val:.4f}±{std:.4f}" for ((cr, val), (cr, std)) in zip(method_avg.items(), method_std.items())])
            else:
                msg = f"{method:18}" + "\t".join([f"{val:.4f}" for (cr, val) in method_avg.items()])
            print(msg)
        else:
            msg = f"{'FairPCA':18}"
            for i in range(len(method_sims)):
                msg += f"eta={method_sims[i]['test']['params']['eta']:.2f}\t"
            print(msg)
    elif method == 'CFPCA':
        if show_err:
            if show_std:
                msg = f"{method:18}" + "\t".join([f"{val:.4f}±{std:.4f}" for ((cr, val), (cr, std)) in zip(method_avg.items(), method_std.items())])
            else:
                msg = f"{method:18}" + "\t".join([f"{val:.4f}" for (cr, val) in method_avg.items()])
            print(msg)
        else:
            msg = f"{'CFPCA':18}"
            for i in range(len(method_sims)):
                msg += f"deta={method_sims[i]['test']['params']['delta']:.2f}\t"
            print(msg)
    elif method == 'CFPCA_LinCov':
        if show_err:
            if show_std:
                msg = f"{method:18}" + "\t".join([f"{val:.4f}±{std:.4f}" for ((cr, val), (cr, std)) in zip(method_avg.items(), method_std.items())])
            else:
                msg = f"{method:18}" + "\t".join([f"{val:.4f}" for (cr, val) in method_avg.items()])
            print(msg)
        else:
            msg = f"{'CFPCA_LinCov':18}"
            for i in range(len(method_sims)):
                msg += f"deta={method_sims[i]['test']['params']['delta']:.2f}\t"
            print(msg)
    elif method == 'PCA':
        if show_err:
            if show_std:
                msg = f"{method:18}" + "\t".join([f"{val:.4f}±{std:.4f}" for ((cr, val), (cr, std)) in zip(method_avg.items(), method_std.items())])
            else:
                msg = f"{method:18}" + "\t".join([f"{val:.4f}" for (cr, val) in method_avg.items()])
            print(msg)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_names', type=str, nargs='*')
    parser.add_argument('--res_dir', type=str, default='exp_results/exp_cv_2loss_k3')
    parser.add_argument('--show_params', action="store_true")
    parser.add_argument('--prefix', type=str, default=None)
    args = parser.parse_args()

    datasets = args.dataset_names
    prefix = args.prefix
    exp_results_folder = args.res_dir
    show_err = not args.show_params
    train_percent = 0.3
    seeds = np.arange(10)

    method_names = ['RFPCA', 'FairPCA', 'CFPCA', 'CFPCA_LinCov']
    # method_names = ['RFPCA', 'FairPCA']
    cnt = {mn: {} for mn in method_names}

    methods_datasets_sims = {method: {} for method in method_names} # params search

    for dataset in datasets:
        # print(f"\nDataset: {dataset}")
        print() 
        out_dir = Path(f'images/{dataset}')
        if prefix is not None:
            out_dir = out_dir.joinpath(prefix)

        out_dir.mkdir(parents=True, exist_ok=True)

        methods_sims = {}
        for method in method_names:
            method_sims = []
            for seed in seeds:
                filepath = glob.glob(f'{exp_results_folder}/{dataset}/{method}_results_{dataset}_trainperc{train_percent}_seed{seed}*.pkl')[0]
                with open(filepath, 'rb') as f:
                    sim = pickle.load(f)
                    method_sims.append(sim)
            assert len(method_sims) == len(seeds)
            methods_sims[method] = method_sims
        
            methods_datasets_sims[method][dataset] = method_sims

        # minlen = min([len(rfpca), len(fpca), len(cfpca), len(pca)])
        # methods_sims = {method: [minfo[i] for i in seed_idx] for method, minfo in methods_sims.items()}

        # criteria = list(method_sims[0]['test']['error'].keys())
        # criteria = {'ABtrain', 'Mtrain'}
        criteria = {'Linear'}
        methods_avg = {}
        methods_std = {}

        for midx, method in enumerate(method_names):
            method_sims= methods_sims[method]
            method_avg = {}
            method_std = {}
            for criterion in criteria:
                method_avg[criterion] = np.mean([method_sims[i]['test']['error'][criterion] for i in range(len(method_sims))], axis=0)
                method_std[criterion] = np.std([method_sims[i]['test']['error'][criterion] for i in range(len(method_sims))], axis=0)
            
            # method_avg['MaxAteBte'] = np.max([method_avg['Atest'], method_avg['Btest']])
            methods_avg[method] = method_avg
            # if midx == 0:
            #     if show_err:
            #         msg = f"{dataset:18}" + "\t".join([f"{cr}" for cr, val in method_avg.items()])
            #         print(msg)
            #     else:
            #         msg = f"{dataset:18}" + "\t\t".join([f"{i}" for i in range(len(method_sims))])
            #         print(msg)

            # dataset_block_present(method, dataset, method_avg, method_std, method_sims, show_err)
            
        dataset_col_present(method_names, dataset, methods_avg)
        
        criteria = methods_avg[method_names[0]].keys()

        for criterion in criteria:
            criterion_info = [methods_avg[mn][criterion] for mn in method_names]
            min_val = min(criterion_info)

            for mn in method_names:
                if criterion not in cnt[mn]:
                    cnt[mn][criterion] = 0

                if methods_avg[mn][criterion] == min_val:
                    cnt[mn][criterion] += 1

    print("Statistics for all datasets ...")
    msg = f'{"Min":15}' + ''.join([f"{cr:10}" for cr, val in cnt[method_names[0]].items()])
    print(msg)
    for method in method_names:
        msg = f'{method:10}' + ''.join([f"{val:10}" for cr, val in cnt[method].items()])
        print(msg)

