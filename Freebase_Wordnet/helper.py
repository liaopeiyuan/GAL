import numpy as np, sys, os, random, pdb, json, uuid, time, argparse
from sklearn.metrics import roc_auc_score, f1_score
from pprint import pprint
from torch import nn
import logging, logging.config
from collections import defaultdict as ddict
from ordered_set import OrderedSet
from collections import Counter
import pandas as pd
from tqdm import tqdm
import gc
import os
import copy

# PyTorch related imports
import torch
from torch.nn import functional as F
from torch.nn.init import xavier_normal_
from torch.utils.data import DataLoader
from torch.nn import Parameter
from torch_scatter import scatter_add

np.set_printoptions(precision=4)

def parse_line(line):
    lhs, attr = line.strip('\n').split('\t')
    return lhs, attr

def parse_file(lines):
    parsed = []
    for line in lines:
        lhs, attr = parse_line(line)
        parsed += [[lhs, attr]]
    return parsed

def get_idx_dicts(data):
    ent_set, attr_set = set(), set()
    for ent, attr in data:
        ent_set.add(ent)
        attr_set.add(attr)
    ent_list = sorted(list(ent_set))
    attr_list = sorted(list(attr_set))

    ent_to_idx, attr_to_idx = {}, {}
    for i, ent in enumerate(ent_list):
        ent_to_idx[ent] = i
    for j, attr in enumerate(attr_list):
        attr_to_idx[attr] = j
    return ent_to_idx, attr_to_idx

def count_attributes(data, attr_to_idx):
    dataset = []
    for ent, attr in data:
        dataset += [attr_to_idx[attr]]
    counts = Counter(dataset)
    return counts

def reindex_attributes(count_list):
    reindex_attr_idx = {}
    for i, attr_tup in enumerate(count_list):
        attr_idx, count = attr_tup[0], attr_tup[1]
        reindex_attr_idx[attr_idx] = i
    return reindex_attr_idx

def transform_data(data, ent_to_idx, attr_to_idx, \
        reindex_attr_idx, attribute_mat):
    dataset = []
    for ent, attr in data:
        attr_idx = attr_to_idx[attr]
        try:
            reidx = reindex_attr_idx[attr_idx]
            attribute_mat[ent_to_idx[ent]][reidx] = 1
        except:
            pass
    return attribute_mat

def set_gpu(gpus):
	"""
	Sets the GPU to be used for the run

	Parameters
	----------
	gpus:           List of GPUs to be used for the run
	
	Returns
	-------
		
	"""
	os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
	os.environ["CUDA_VISIBLE_DEVICES"] = gpus

def get_logger(name, log_dir, config_dir):
	"""
	Creates a logger object

	Parameters
	----------
	name:           Name of the logger file
	log_dir:        Directory where logger file needs to be stored
	config_dir:     Directory from where log_config.json needs to be read
	
	Returns
	-------
	A logger object which writes to both file and stdout
		
	"""
	config_dict = json.load(open( config_dir + 'log_config.json'))
	config_dict['handlers']['file_handler']['filename'] = os.path.join(log_dir, name.replace('/', '-'))
	logging.config.dictConfig(config_dict)
	logger = logging.getLogger(name)

	std_out_format = '%(asctime)s - [%(levelname)s] - %(message)s'
	consoleHandler = logging.StreamHandler(sys.stdout)
	consoleHandler.setFormatter(logging.Formatter(std_out_format))
	logger.addHandler(consoleHandler)

	return logger

def get_combined_results(left_results, right_results):
	results = {}
	count   = float(left_results['count'])

	results['left_mr']	= round(left_results ['mr'] /count, 5)
	results['left_mrr']	= round(left_results ['mrr']/count, 5)
	results['right_mr']	= round(right_results['mr'] /count, 5)
	results['right_mrr']	= round(right_results['mrr']/count, 5)
	results['mr']		= round((left_results['mr']  + right_results['mr']) /(2*count), 5)
	results['mrr']		= round((left_results['mrr'] + right_results['mrr'])/(2*count), 5)

	for k in range(10):
		results['left_hits@{}'.format(k+1)]	= round(left_results ['hits@{}'.format(k+1)]/count, 5)
		results['right_hits@{}'.format(k+1)]	= round(right_results['hits@{}'.format(k+1)]/count, 5)
		results['hits@{}'.format(k+1)]		= round((left_results['hits@{}'.format(k+1)] + right_results['hits@{}'.format(k+1)])/(2*count), 5)
	return results

def get_param(shape):
	param = Parameter(torch.Tensor(*shape)); 	
	xavier_normal_(param.data)
	return param

def com_mult(a, b):
	r1, i1 = a[..., 0], a[..., 1]
	r2, i2 = b[..., 0], b[..., 1]
	return torch.stack([r1 * r2 - i1 * i2, r1 * i2 + i1 * r2], dim = -1)

def conj(a):
	a[..., 1] = -a[..., 1]
	return a

def cconv(a, b):
	return torch.irfft(com_mult(torch.rfft(a, 1), torch.rfft(b, 1)), 1, signal_sizes=(a.shape[-1],))

def ccorr(a, b):
	return torch.irfft(com_mult(conj(torch.rfft(a, 1)), torch.rfft(b, 1)), 1, signal_sizes=(a.shape[-1],))


class GradReverse(torch.autograd.Function):
    """
    Implement the gradient reversal layer for the convenience of domain adaptation neural network.
    The forward part is the identity function while the backward part is the negative function.
    """
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()
		
class GradientReversalLayer(nn.Module):
    def __init__(self):
        super(GradientReversalLayer, self).__init__()

    def forward(self, inputs):
        return GradReverse.apply(inputs)