import os, sys
import numpy as np 
import csv
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import math 
import pandas as pd 
import pickle
from pathlib import Path
import os.path as osp
import shutil
import tempfile
import platform
from importlib import import_module
import ast
import glob 
import argparse
from itertools import product
from data_loaders import DATASETS

def make_param_grid(d):
    '''
    Param: 
        d: a dict of param, e.g. {'a': [1,2], 'b': [3,4]}
    Return
        grid: list of config, e.g. [{'a': 1, 'b': 3}, 
                                    {'a': 1, 'b': 4}, 
                                    {'a': 2, 'b': 3}, 
                                    {'a': 2, 'b': 4}]
    '''
    iter_params = {name: val for name, val in d.items() if isinstance(val, list) or isinstance(val, np.ndarray)}
    non_iter_params = {name: val for name, val in d.items() if not (isinstance(val, list) or isinstance(val, np.ndarray))}

    grid = []
    for vcomb in product(*iter_params.values()):
        grid.append(dict(zip(iter_params.keys(), vcomb)))

    for params in grid:
        params.update(non_iter_params)

    return np.array(grid, dtype='object')

class Tee:
    def __init__(self, fname, mode="a"):
        self.stdout = sys.stdout
        self.file = open(fname, mode)

    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)
        self.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()


def compute_criteria(M, M_pr, A, A_pr, B, B_pr):
    reconsError_M = np.linalg.norm(M - M_pr, 'fro')**2 / M.shape[0]
    reconsError_A = np.linalg.norm(A - A_pr, 'fro')**2 / A.shape[0]
    reconsError_B = np.linalg.norm(B - B_pr, 'fro')**2 / B.shape[0]
    reconsError_AB = abs(reconsError_A - reconsError_B)
    return {'reM': reconsError_M, 'reA': reconsError_A, 
            'reB': reconsError_B, 'reAB': reconsError_AB}

def compute_errors(Mtrain, Atrain, Btrain, Strain, Ytrain, Mtest, Atest, Btest, Stest, Ytest, ProjMat, criteria):
    errors = {}
    compute_train_err = False
    compute_test_err = False

    for criterion in criteria:
        if 'train' in criterion:
            compute_train_err = True
        if 'test' in criterion:
            compute_test_err = True

    # In-sample evaluation
    if compute_train_err:
        Mtrain_pr = Mtrain@ProjMat
        Atrain_pr = Atrain@ProjMat
        Btrain_pr = Btrain@ProjMat

        insample_criteria_values = compute_criteria(Mtrain, Mtrain_pr, 
                                                    Atrain, Atrain_pr, Btrain, Btrain_pr)
        
        for criterion in criteria:
            if 'train' in criterion:
                map_name = 're' + criterion.replace('train', '')
                errors[criterion] = insample_criteria_values[map_name]

    if compute_test_err:
        # Out-sample evaluation
        Mtest_pr = Mtest@ProjMat
        Atest_pr = Atest@ProjMat
        Btest_pr = Btest@ProjMat

        outsample_criteria_values = compute_criteria(Mtest, Mtest_pr, 
                                                    Atest, Atest_pr, Btest, Btest_pr)
        for criterion in criteria:
            if 'test' in criterion:
                map_name = 're' + criterion.replace('test', '')
                errors[criterion] = outsample_criteria_values[map_name]

    return errors

def plot_pareto(errors_info, partitions=['train', 'test'], out_dir='images/', dataset='biodeg'):
    K = errors_info['RFPCA']['params']['K']
    d = errors_info['RFPCA']['params']['d']
    color = np.array(['r', 'b', 'm', 'g', 'c', 'y', 'k', 'w'])
    marker = ["o", "v", "P", "s", "*"]
    cm = plt.cm.get_cmap('RdYlBu')
    
    methods_name = errors_info.keys()

    for data_set in partitions:
        name = 'InSample' if data_set == 'train' else 'OutSample'
        for k in K: # Pareto plots
            fig = plt.figure(figsize = (10,10), dpi = 200)
            matplotlib.rcParams.update({'font.size': 18})
            ax = fig.add_subplot(111)

            for idx, (method_name, method_info) in enumerate(errors_info.items()):
                method_Mtrain, method_ABtrain = method_info['error'][k][f'M{data_set}'], method_info['error'][k][f'AB{data_set}']
                if 'FairPCA' in method_name:
                    sc = ax.scatter(method_Mtrain.flatten(), method_ABtrain.flatten(), marker=marker[idx], s=160, cmap=cm, label=method_name)
                elif 'RFPCA' in method_name:
                   sc = ax.scatter(method_Mtrain.flatten(), method_ABtrain.flatten(), marker=marker[idx], s=200, cmap=cm, label=method_name)
                else:
                    ax.scatter(method_Mtrain.flatten(), method_ABtrain.flatten(), marker=marker[idx], s=200, c = color[idx], label=method_name)
            # ax.set_title(f'Pareto plot with {k} features - \n{name} ({DATASETS[dataset]["name"]} {d}-features)')
            ax.set_xlabel('Reconstruction error (all data)')
            ax.set_ylabel('Absolute difference of reconstruction error between groups')
            ax.set_xscale('log')
            ax.legend(fontsize=22)
            
            plt.savefig(f'{out_dir}/pareto_{name}_{k}features_{dataset}.png', dpi=200, bbox_inches='tight')
            plt.close()
            print('saved to ', f'{out_dir}/pareto_{name}_{k}features_{dataset}.png')

def load_config_from_file(filename):
    def check_file_exist(filename, msg_tmpl='file "{}" does not exist'):
        if not osp.isfile(filename):
            raise FileNotFoundError(msg_tmpl.format(filename))

    def _validate_py_syntax(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            # Setting encoding explicitly to resolve coding issue on windows
            content = f.read()
        try:
            ast.parse(content)
        except SyntaxError as e:
            raise SyntaxError('There are syntax errors in config '
                              f'file {filename}: {e}')

    filename = osp.abspath(osp.expanduser(filename))
    check_file_exist(filename)
    fileExtname = osp.splitext(filename)[1]
    if fileExtname not in ['.py', '.json', '.yaml', '.yml']:
        raise IOError('Only py/yml/yaml/json type are supported now!')

    with tempfile.TemporaryDirectory(dir='logs') as temp_config_dir:
        with tempfile.NamedTemporaryFile(dir=temp_config_dir, 
                suffix=fileExtname) as temp_config_file:

            if platform.system() == 'Windows':
                temp_config_file.close()
            temp_config_name = osp.basename(temp_config_file.name)
            shutil.copyfile(filename, temp_config_file.name)

            if filename.endswith('.py'):
                temp_module_name = osp.splitext(temp_config_name)[0]
                sys.path.insert(0, temp_config_dir)
                _validate_py_syntax(filename)
                mod = import_module(temp_module_name)
                sys.path.pop(0)
                cfg_dict = {
                    name: value
                    for name, value in mod.__dict__.items()
                    if not name.startswith('__')
                }
                # delete imported module
                del sys.modules[temp_module_name]

    return cfg_dict

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_name', type=str)
    parser.add_argument('--res_dir', type=str, default='exp_results/pareto_curve_insample')
    parser.add_argument('--prefix', type=str, default=None)
    args = parser.parse_args()

    dataset = args.dataset_name
    prefix = args.prefix
    exp_results_folder = args.res_dir
    train_percent = 1.0

    out_dir = Path(f'images/pareto/{dataset}')
    if prefix is not None:
        out_dir = out_dir.joinpath(prefix)

    out_dir.mkdir(parents=True, exist_ok=True)

    criteria = [
        'ABtrain', 
        'Mtrain',
        # 'ABtest',
        # 'Mtest'
    ]
    examined_methods = ['RFPCA', 'FairPCA', 'CFPCA']
    avg_methods = {}

    for method in examined_methods:
        run_id = f"{method}_results_{dataset}_trainperc{train_percent}_seed*"

        method_sims = []
        for filepath in glob.glob(f'{exp_results_folder}/{dataset}/{run_id}.pkl'):
            with open(filepath, 'rb') as f:
                sim = pickle.load(f)
                method_sims.append(sim)
        assert len(method_sims) >= 5

        K = list(set([p['K'] for p in method_sims[0]['params']]))
        d = method_sims[0]['params'][0]['d']
        params = {}

        if method == 'RFPCA':
            alpha = list(set([p['alpha'] for p in method_sims[0]['params']]))
            lamda = list(set([p['lamda'] for p in method_sims[0]['params']]))
            params['alpha'] = alpha
            params['lamda'] = lamda

        elif method =='FairPCA':
            lamda = list(set([p['eta'] for p in method_sims[0]['params']]))
            params['eta'] = lamda

        elif 'CFPCA' in method:
            params['mu'] = list(set([p['mu'] for p in method_sims[0]['params']]))
            params['delta'] = list(set([p['delta'] for p in method_sims[0]['params']]))

        avg_method = {'error': {}, 'params': params}
        avg_method['params']['K'] = K
        avg_method['params']['d'] = d

        for k in K:
            avg_method['error'][k] = {}
            for key in criteria:
                avg = np.mean(np.array([method_sims[i]['test'][key] for i in range(len(method_sims))]), axis=0)
                avg_method['error'][k][key] = avg

        avg_methods[method] = avg_method
        
    # fpca = fpca['error']
    # rfpca = rfpca['error']
    plot_pareto(avg_methods, partitions=['train'], out_dir=str(out_dir), dataset=dataset)
    # plot_param_loss_at_alphas(rfpca, fpca, loss='ABtrain', name='ABtrain')

