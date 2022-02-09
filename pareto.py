import os, sys, time, random, datetime
import pickle
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
from RFPCA import _fit_RFPCA
from FairPCA import _fit_FairPCA
from CFPCA import _fit_CFPCA, run_SVM

from functions import * 
from data_loaders import load_data_olfat, normalize
import argparse
# from Linear_Ferm_SVM import run_FERM
from pathlib import Path

from sklearn.model_selection import KFold

def crossval(Xtrain, Strain, Ytrain, params_grid, criteria, method_fit_func, k_fold=3):

    kf = KFold(n_splits=k_fold, random_state=manualSeed, shuffle=True)
    crossval_errors = []

    for fold_idx, (train_index, val_index) in enumerate(kf.split(Xtrain)):
        print(f"\nRunning CV Fold {fold_idx} ...")

        Xtrain_fd, Strain_fd, Ytrain_fd = (ss[train_index].copy() for ss in [Xtrain, Strain, Ytrain])
        Xval_fd, Sval_fd, Yval_fd = (ss[val_index].copy() for ss in [Xtrain, Strain, Ytrain])

        Atr = Xtrain_fd[Strain_fd == 0].copy()
        Btr = Xtrain_fd[Strain_fd == 1].copy()

        Ava = Xval_fd[Sval_fd == 0].copy()
        Bva = Xval_fd[Sval_fd == 1].copy()

        # Normalize
        Xtrain_fd, meanXtr, stdXtr = normalize(Xtrain_fd)
        Xval_fd= normalize(Xval_fd, meanXtr, stdXtr)[0]
        
        Atr, meanAtr, _ = normalize(Atr, scale=False)
        Ava = normalize(Ava, meanAtr, scale=False)[0]

        Btr, meanBtr, _ = normalize(Btr, scale=False)
        Bva = normalize(Bva, meanBtr, scale=False)[0]
        
        fold_errors = {criterion: np.zeros(len(params_grid)) - 1e6 for criterion in criteria}

        for idx, params in enumerate(params_grid):
            
            V, PrV = method_fit_func(Xtrain_fd, Atr, Btr, Strain_fd, Ytrain_fd, params)

            err = compute_errors(Xtrain_fd, Atr, Btr, Strain_fd, Ytrain_fd, 
                                 Xval_fd, Ava, Bva, Sval_fd, Yval_fd, PrV, criteria)

            for criterion in criteria:
                fold_errors[criterion][idx] = err[criterion]
        
        crossval_errors.append(fold_errors)

    return crossval_errors

def _fit_PCA(Xtr, Atr, Btr, Str, Ytr, params):
    k = params['K']
    # eigen decomposition
    eigs, vecs = np.linalg.eigh(1./Xtr.shape[0] * Xtr.T@Xtr)
    V = vecs[:, -k:]
    ProjMat = V@V.T
    assert ProjMat.shape[0] == Xtr.shape[1] and ProjMat.shape[1] == Xtr.shape[1]

    # from sklearn.decomposition import PCA
    # pca = PCA(n_components=k)
    # pca.fit(Mtrain)
    # vecs_pca = pca.components_.T
    # ProjMat_pca = vecs_pca@vecs_pca.T
    # reconsError_pca = compute_errors(Mtrain, Atrain, Btrain, Strain, Ytrain, Mtest, Atest, Btest, ProjMat_pca)
    # print(reconsError)
    return V, ProjMat


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Simulation")
    parser.add_argument('--result_folder', type=str, default='exp_results', help='root folder for results and stuffs from training process')
    parser.add_argument('--method', type=str, default='RFPCA', help='method to use, RFPCA/FairPCA')
    parser.add_argument('--config', type=str, default='configs/RFPCA.py', help='config path to use')
    parser.add_argument('--dataset', type=str, default='credit', help='data file path')
    parser.add_argument('--file_path', type=str, default=None, help='data file path')
    parser.add_argument('--train_percent', type=float, default=0.5, help='proportion of data for training, others for testing')
    parser.add_argument('--run_cv', action='store_true')
    parser.add_argument('--k_fold', type=int, default=3)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.seed is None:
        manualSeed = np.random.randint(10000000)
    else:
        manualSeed = args.seed

    np.random.seed(manualSeed)
    random.seed(manualSeed)

    filepath = args.file_path
    train_percent = args.train_percent
    dataset = args.dataset
    method = args.method
    run_CV = args.run_cv
    k_fold = args.k_fold

    from data_loaders import NAMES
    if dataset not in NAMES:
        print(f"dataset '{dataset}' is not supported!")
        sys.exit(1)

    RESULT_FOLDER = Path(args.result_folder)
    RESULT_FOLDER = RESULT_FOLDER.joinpath(f'{dataset}')
    RESULT_FOLDER.mkdir(parents=True, exist_ok=True)

    now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    run_id = f"{method}_results_{dataset}_trainperc{train_percent}_seed{manualSeed}_{now_str}"
    logfile_path = log_dir / f'{run_id}.txt'
    
    sys.stdout = Tee(logfile_path, "a")
    print(f'saving logs to {logfile_path}')
    
    # fair reconstruction error
    criteria = [
        'Atrain', 
        'Btrain', 
        'ABtrain', 
        'Mtrain',
        'Atest', 
        'Btest', 
        'ABtest',
        'Mtest'
    ]

    hyparams = load_config_from_file(args.config)['hyper_params']
    K = hyparams['K']

    train_info, test_info = load_data_olfat(dataset, train_percent, random_seed=manualSeed)
    # Induced params
    d = train_info[0].shape[1] ## no. of features
    hyparams['d'] = d
    K = [k for k in K if k < d]
    hyparams['K'] = K

    tic = time.time()
    print(f'Simulations starts!')

    if method == 'RFPCA':
        fit_func = _fit_RFPCA
    elif method == 'FairPCA':
        fit_func = _fit_FairPCA
    elif method == 'CFPCA' or method == 'CFPCA_LinCov':
        fit_func = _fit_CFPCA
    elif method == 'PCA':
        fit_func = _fit_PCA
    else:
        print(f"method {method} is not supported!")
        sys.exit(0)

    Xtrain, Strain, Ytrain = train_info
    Xtest, Stest, Ytest = test_info

    val_criteria = ['Mtest', 'ABtest']

    grid_params = make_param_grid(hyparams)

    if run_CV:
        if len(grid_params) > 1:
            crossval_errors = crossval(Xtrain, Strain, Ytrain, grid_params, 
                                            val_criteria, fit_func, k_fold=k_fold)
            print()
            sum_val_criterion = [{"val_obj": np.sum([crossval_errors[i][cr] for cr in val_criteria], axis=0)} for i in range(len(crossval_errors))]
            avg_val_criterion = np.mean([sum_val_criterion[i]['val_obj'] for i in range(len(sum_val_criterion))], axis=0) 
            assert len(avg_val_criterion) == len(grid_params)
            best_params = grid_params[avg_val_criterion == avg_val_criterion.min()]
            if len(best_params) > 1: print("there are more than 1 optimal params")
            best_params = best_params[0]
            print(f"best params: {best_params} with avg ({val_criteria})={avg_val_criterion.min()}")
        else:
            crossval_errors = []
            best_params = grid_params[0]
            print(f"one params: {best_params}")


    # Final training with best params and get results
    Atr = Xtrain[Strain == 0].copy()
    Atr, meanAtr, _ = normalize(Atr, scale=False)
        
    Btr = Xtrain[Strain == 1].copy()
    Btr, meanBtr, _ = normalize(Btr, scale=False)


    if Xtest is not None:
        Ate = Xtest[Stest == 0].copy()
        Ate = normalize(Ate, meanAtr, scale=False)[0]

        Bte = Xtest[Stest == 1].copy()
        Bte = normalize(Bte, meanBtr, scale=False)[0]
    else:
        Ate, Bte = None, None
        criteria = {cr for cr in criteria if 'test' not in cr}

    if run_CV:
        V, PrV = fit_func(Xtrain, Atr, Btr, Strain, Ytrain, best_params)

        reconsError = compute_errors(Xtrain, Atr, Btr, Strain, Ytrain, 
                        Xtest, Ate, Bte, Stest, Ytest, PrV, criteria)

        varExpTr = np.trace(V.T@Xtrain.T@Xtrain@V) # (K, D) * (D, D) * (D, K)
        totVarTr = np.trace(Xtrain.T@Xtrain)
        reconsError['varExpTr'] = varExpTr/totVarTr

        if Xtest is not None:
            varExp = np.trace(V.T@Xtest.T@Xtest@V) # (K, D) * (D, D) * (D, K)
            totVar = np.trace(Xtest.T@Xtest)
            reconsError['varExpTe'] = varExp/totVar
            print()

            svm_error = run_SVM(Xtest, Stest, V, norm_func=normalize)
            reconsError.update(svm_error)

            msg = "\t".join([f"{cr}" for cr, val in reconsError.items()]) + "\n"
            msg += "\t".join([f"{val:.4f}" for cr, val in reconsError.items()])
            print(msg)
    else:
        reconsError = {criterion: np.zeros(len(grid_params)) for criterion in criteria}

        for param_idx, curr_params in enumerate(grid_params):
            print(f"\nHy-params: {curr_params}")
            V, PrV = fit_func(Xtrain, Atr, Btr, Strain, Ytrain, curr_params)
            print()
            reconsError_param = compute_errors(Xtrain, Atr, Btr, Strain, Ytrain, 
                            Xtest, Ate, Bte, Stest, Ytest, PrV, criteria)

            varExpTr = np.trace(V.T@Xtrain.T@Xtrain@V) # (K, D) * (D, D) * (D, K)
            totVarTr = np.trace(Xtrain.T@Xtrain)
            reconsError_param['varExpTr'] = varExpTr/totVarTr

            if Xtest is not None:
                # Xtest = (np.eye(Xtest.shape[0])-np.ones((Xtest.shape[0], Xtest.shape[0]))/Xtest.shape[0]) @ Xtest
                varExp = np.trace(V.T@Xtest.T@Xtest@V) # (K, D) * (D, D) * (D, K)
                totVar = np.trace(Xtest.T@Xtest)
                reconsError_param['varExpTe'] = varExp/totVar

                svm_error = run_SVM(Xtest, Stest, V, norm_func=normalize)
                reconsError_param.update(svm_error)

                msg = "\t".join([f"{cr}" for cr, val in reconsError_param.items()]) + "\n"
                msg += "\t".join([f"{val:.4f}" for cr, val in reconsError_param.items()])
                print(msg)
            
            for criterion in reconsError_param.keys():
                if criterion not in reconsError:
                    reconsError[criterion] = np.zeros(len(grid_params))
                    
                reconsError[criterion][param_idx] = reconsError_param[criterion]


    toc = time.time() - tic
    print(f'Simulations are done in {toc:.2f} sec!\n')

    savepath = str(RESULT_FOLDER) + f'/{run_id}.pkl'
    with open(savepath, 'wb') as f:
        print(f"saving results to {savepath}...")
        # saveobj = (reconsError, clf_criteria)
        if run_CV:
            saveobj = {'CV': {'error': crossval_errors, 'params_grid': grid_params},  
                    'test': {'error': reconsError, 'params': best_params}}
        else:
            saveobj = {'test': reconsError, 'params': grid_params}
        pickle.dump(saveobj, f)

