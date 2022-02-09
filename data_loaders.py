import os, sys
import numpy as np 
import csv
from matplotlib import pyplot as plt
import math 
import pandas as pd 
import pickle

NAMES = ['default_credit', 'biodeg', 'e_coli', 'energy', 'german_credit', 'image', 
		'letter', 'magic', 'parkinson',  'skillcraft', 
		'statlog', 'steel', 'taiwan_credit', 'wine_quality']

DATASETS = {
	"default_credit": {
		"name": "Default Credit",
		"path": "data/default_credit.csv"
	},

	"biodeg": {
		"name": "Biodeg",
		"path": "data/Biodeg_Data.csv"
	},
	"e_coli": {
		"name": "E. Coli",
		"path": "data/Ecoli_Data.csv"
	},
	"energy": {
		"name": "Energy",
		"path": "data/Energy_Data.csv"
	},
	"german_credit": {
		"name": "German Credit",
		"path": "data/German_Credit_Data.csv"
	},
	"image": {
		"name": "Image",
		"path": "data/Image_Seg_Data.csv"
	},
	"letter": {
		"name": "Letter",
		"path": "data/Letter_Rec_Data.csv"
	},
	"magic": {
		"name": "Magic",
		"path": "data/Magic_Data.csv"
	},
	"parkinson": {
		"name": "Parkinson's",
		"path": "data/Parkinsons_Data.csv"
	},
	"skillcraft": {
		"name": "SkillCraft",
		"path": "data/SkillCraft_Data.csv"
	},
	"statlog": {
		"name": "Statlog",
		"path": "data/Statlog_Data.csv"
	},
	"steel": {
		"name": "Steel",
		"path": "data/Steel_Data.csv"
	},
	"taiwan_credit": {
		"name": "Taiwan Credit",
		"path": "data/Taiwanese_Credit_Data.csv"
	},
	"wine_quality": {
		"name": "Wine Quality",
		"path": "data/Wine_Quality_Data.csv"
	}
}

def stratified_split_S_based(X, S, Y, perc=0, rng=None, standardize=False): 
	if perc == 0:
		if standardize:
			print("Standard scaling ...")
			X = normalize(X)[0]
		return (None, None, None), (X, S, Y)
	elif perc == 1.0:
		if standardize:
			print("Standard scaling ...")
			X = normalize(X)[0]
		return (X, S, Y), (None, None, None)

	if isinstance(rng, int):
		rng = np.random.default_rng(rng)

	assert len(X) == len(S)

	group0_idx = np.arange(len(X))[S == 0]
	group1_idx = np.arange(len(X))[S == 1]
	N0 = len(group0_idx)
	N1 = len(group1_idx)

	rng.shuffle(group0_idx)
	trainidx_g0 = group0_idx[:int(perc*N0)]
	testidx_g0 = group0_idx[int(perc*N0):]    

	rng.shuffle(group1_idx)
	trainidx_g1 = group1_idx[:int(perc*N1)]
	testidx_g1 = group1_idx[int(perc*N1):]    
	
	trainidx = np.concatenate([trainidx_g0, trainidx_g1], axis=0)
	testidx = np.concatenate([testidx_g0, testidx_g1], axis=0)
	rng.shuffle(trainidx)
	rng.shuffle(testidx)

	if standardize:
		print("Standard scaling ...")
		X[trainidx], mean, std = normalize(X[trainidx])
		X[testidx] = normalize(X[testidx], mean, std)[0]
	
	train_info = X[trainidx], S[trainidx], Y[trainidx]
	test_info = X[testidx], S[testidx], Y[testidx]

	return train_info, test_info

def normalize(X, mean=None, std=None, scale=True):
	mean = mean if mean is not None else X.mean(axis=0, keepdims=True)
	X = X - mean
	if scale:
		std = std if std is not None else X.std(axis=0, keepdims=True)
		std[np.where(std == 0)] = 1
		X = X / std
	return X, mean, std

def load_data_olfat(dataset, train_percent=0.3, random_seed=None, standardize=True):
	if dataset not in DATASETS:
			print(f"dataset name '{dataset}' is not supported!")
			return None
			
	filepath = DATASETS[dataset]['path']

	if random_seed is None:
		# print("Loading data with random seed ...")
		rng = np.random
	else:
		# print(f"Loading data with manual seed {random_seed} ...")
		rng = np.random.default_rng(random_seed)

	df = pd.read_csv(filepath, skipinitialspace=True)
	stds = df.std(axis=0)
	dropcols = stds[stds==0].index.tolist()
	if dataset == 'steel':
		dropcols += ['perimX', 'perimY']

	df = df.drop(columns = dropcols)
	headers = df.columns
	print("headers: \n", headers.to_numpy())
	print(f"Prediction class: '{headers[-2]}', Sensitive attribute: '{headers[-1]}'")
	data = df.to_numpy()
	Y = data[:,-2].copy().astype(int) # prediction class
	S = data[:,-1].copy().astype(int) # protected/sensitive attribute
	data = data[:,:-2].copy().astype(float)
	print(f"Data used: {dataset}: {data.shape}")
	# convert label to -1, 1
	Y[Y == 0] = 1

	train_info, test_info = stratified_split_S_based(data, S, Y, train_percent, rng, standardize=standardize)
	print(f"Train {train_info[0].shape[0]}, Test: {test_info[0].shape[0] if test_info[0] is not None else 0}")

	return train_info, test_info

if __name__ == '__main__':
	# modify_default_credit('../new/data/default_credit/UCI_Credit_Card.csv')
	load_data_olfat('default_credit', train_percent=0.5)