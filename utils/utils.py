"""Utilities for training and testing
"""
# MIT License
# 
# Copyright (c) 2019 Yichun Shi
# Copyright (c) 2021 Kaen Chan
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import sys
import os
import numpy as np
from scipy import misc
import time
import math
import random
from datetime import datetime
import shutil

from abc import ABCMeta
from six import with_metaclass

def create_log_dir(config, config_file):
    subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    log_dir = os.path.join(os.path.expanduser(config.log_base_dir), config.name, subdir)
    if not os.path.isdir(log_dir):  # Create the log directory if it doesn't exist
        os.makedirs(log_dir)
    shutil.copyfile(config_file, os.path.join(log_dir,'config.py'))

    return log_dir


def get_updated_learning_rate(global_step, config):
    if config.learning_rate_strategy == 'step':
        max_step = -1
        learning_rate = 0.0
        for step, lr in config.learning_rate_schedule.items():
            if global_step >= step and step > max_step:
                learning_rate = lr
                max_step = step
        if max_step == -1:
            raise ValueError('cannot find learning rate for step %d' % global_step)
    elif config.learning_rate_strategy == 'cosine':
        initial = config.learning_rate_schedule['initial']
        interval = config.learning_rate_schedule['interval']
        end_step = config.learning_rate_schedule['end_step']
        step = math.floor(float(global_step) / interval) * interval
        assert step <= end_step
        learning_rate = initial * 0.5 * (math.cos(math.pi * step / end_step) + 1)
    elif config.learning_rate_strategy == 'linear':
        initial = config.learning_rate_schedule['initial']
        start = config.learning_rate_schedule['start']
        end_step = config.learning_rate_schedule['end_step']
        assert global_step <= end_step
        assert start < end_step
        if global_step < start:
            learning_rate = initial
        else:
            learning_rate = 1.0 * initial * (end_step - global_step) / (end_step - start)
    else:
        raise ValueError("Unkown learning rate strategy!")

    return learning_rate

def display_info(epoch, step, duration, watch_list):
    s = '[%d][%d] time: %2.2f' % (epoch+1, step+1, duration)
    keys = list(watch_list.keys())
    keys.sort()
    for key in keys:
        value = watch_list[key]
        item = [key, value]
        if type(item[1]) in [float, np.float32, np.float64]:
            s += ' %s %2.4f' % (item[0], item[1])
        elif type(item[1]) in [int, bool, np.int32, np.int64, np.bool]:
            s += ' %s: %d' % (item[0], item[1])
    return s

def l2_normalize(x, axis=None, eps=1e-8):
    x = x / (eps + np.linalg.norm(x, axis=axis))
    return x

def pair_euc_score(x1, x2):
    x1, x2 = np.array(x1), np.array(x2)
    dist = np.sum(np.square(x1 - x2), axis=1)
    return -dist

def pair_cosin_score(x1, x2):
    # x1 = [l2_normalize(l) for l in x1]
    # x2 = [l2_normalize(l) for l in x2]
    t1 = time.time()
    x1 = np.divide(x1.T, np.linalg.norm(x1.T, axis=0)).T
    x2 = np.divide(x2.T, np.linalg.norm(x2.T, axis=0)).T
    # x1, x2 = np.array(x1), np.array(x2)
    score = np.sum(np.multiply(x1, x2), axis=1)
    # print('-->time pair_cosin_score', time.time()-t1)
    return score

def pair_MLS_score(x1, x2, sigma_sq1=None, sigma_sq2=None, use_attention_only=False):
    t1 = time.time()
    if sigma_sq1 is None:
        x1, x2 = np.array(x1), np.array(x2)
        assert sigma_sq2 is None, 'either pass in concated features, or mu, sigma_sq for both!'
        D = int(x1.shape[1] / 2)
        if x1.shape[1] == 257 or x1.shape[1] == 513:
            D = int(x1.shape[1] - 1)
        # D = int(x1.shape[1] - 1)
        mu1, sigma_sq1 = x1[:,:D], x1[:,D:]
        mu2, sigma_sq2 = x2[:,:D], x2[:,D:]
    else:
        x1, x2 = np.array(x1), np.array(x2)
        D = int(x1.shape[1])
        sigma_sq1, sigma_sq2 = np.array(sigma_sq1), np.array(sigma_sq2)
        mu1, mu2 = x1, x2
    if D == x1.shape[1] - 1:
        sigma_sq_mutual = sigma_sq1 + sigma_sq2
        cos_theta = np.sum(mu1*mu2, axis=1)
        sigma_sq_mutual = sigma_sq_mutual.ravel()
        dist1 = 2*(1-cos_theta) / (1e-10 + sigma_sq_mutual)
        dist2 = np.log(sigma_sq_mutual)

        if use_attention_only:
            dist = dist1
        else:
            dist = dist1 + dist2
        return -dist
    sigma_sq_mutual = sigma_sq1 + sigma_sq2
    dist1 = np.mean(np.square(mu1 - mu2) / sigma_sq_mutual, axis=1)
    dist2 = np.mean(np.log(sigma_sq_mutual), axis=1)
    if use_attention_only:
        dist = dist1
    else:
        dist = dist1 + dist2
    # print('-->time pair_MLS_score', time.time()-t1)
    return -dist


def l2_distance_matrix(X1, X2):
    XXh1 = np.sum(np.square(X1), reduction_indices=1)
    XXh2 = np.sum(np.square(X2), reduction_indices=1)
    XXh1 = np.reshape(XXh1, [1,-1])
    XXh2 = np.reshape(XXh2, [1,-1])
    l2dist = np.transpose(XXh1) + XXh2 - 2*np.matmul(X1, np.transpose(X2))
    return l2dist


def l2_distance_matrix(X1, X2, sigma_sq):
    XXh1 = np.sum(np.square(X1), reduction_indices=1)
    XXh2 = np.sum(np.square(X2), reduction_indices=1)
    XXh1 = np.reshape(XXh1, [1,-1])
    XXh2 = np.reshape(XXh2, [1,-1])
    l2dist = np.transpose(XXh1) + XXh2 - 2*np.matmul(X1, np.transpose(X2))
    return l2dist


def nvm_MLS_score(x1, x2, sigma_sq1=None, sigma_sq2=None):
    total, _, _ = nvm_MLS_score_detail(x1, x2, sigma_sq1, sigma_sq2)
    return total


def nvm_MLS_score_attention(x1, x2, sigma_sq1=None, sigma_sq2=None):
    _, dist, _ = nvm_MLS_score_detail(x1, x2, sigma_sq1, sigma_sq2)
    return dist


def nvm_MLS_score_detail(x1, x2, sigma_sq1=None, sigma_sq2=None):
    one_sigma_sq1 = None
    one_sigma_sq2 = None
    if sigma_sq1 is None:
        x1, x2 = np.array(x1), np.array(x2)
        assert sigma_sq2 is None, 'either pass in concated features, or mu, sigma_sq for both!'
        D = int(x1.shape[1] / 2)
        if x1.shape[1] == 257 or x1.shape[1] == 513:
            D = int(x1.shape[1] - 1)
        mu1, sigma_sq1 = x1[:,:D], x1[:,D:]
        mu2, sigma_sq2 = x2[:,:D], x2[:,D:]
        if x1.shape[1] == 513:
            mu1, sigma_sq1, one_sigma_sq1 = x1[:,:D], x1[:,D:2*D], x1[:,2*D:]
            mu2, sigma_sq2, one_sigma_sq2 = x2[:,:D], x2[:,D:2*D], x2[:,2*D:]
    else:
        x1, x2 = np.array(x1), np.array(x2)
        D = int(x1.shape[1])
        sigma_sq1, sigma_sq2 = np.array(sigma_sq1), np.array(sigma_sq2)
        mu1, mu2 = x1, x2

    t1 = time.time()
    if D == x1.shape[1] - 1:
        sigma_sq_mutual = sigma_sq1 + np.transpose(sigma_sq2)
        print('   s1', time.time()-t1)
        cos_theta = np.dot(mu1, mu2.T)
        print('   s2', time.time()-t1)
        dist1 = 2*(1-cos_theta) / (1e-10 + sigma_sq_mutual)
        dist2 = np.log(sigma_sq_mutual)
        print('   s3', time.time()-t1)
        dist = dist1 + dist2
        return -dist, -dist1, -dist2
    # print(x1.shape, x2.shape, mu1.shape, sigma_sq1.shape, mu2.shape, sigma_sq2.shape)
    sigma_sq1 = sigma_sq1.reshape((sigma_sq1.shape[0], 1, sigma_sq1.shape[1]))
    sigma_sq2 = sigma_sq2.reshape((1, sigma_sq2.shape[0], sigma_sq2.shape[1]))
    sigma_sq_mutual = sigma_sq1 + sigma_sq2
    if one_sigma_sq1 is not None:
        one_sigma_sq1 = one_sigma_sq1.reshape((one_sigma_sq1.shape[0], 1, 1))
        one_sigma_sq2 = one_sigma_sq2.reshape((1, one_sigma_sq2.shape[0], 1))
        sigma_sq_mutual += one_sigma_sq1 + one_sigma_sq2
    del sigma_sq1
    del sigma_sq2
    print('   s1', time.time()-t1)
    mu1 = mu1.reshape((mu1.shape[0], 1, mu1.shape[1]))
    mu2 = mu2.reshape((1, mu2.shape[0], mu2.shape[1]))
    # dist = np.sum(np.square(mu1 - mu2) / sigma_sq_mutual + np.log(sigma_sq_mutual), axis=1)
    dist1 = np.mean(np.square(mu1 - mu2) / sigma_sq_mutual, axis=-1)
    dist2 = np.mean(np.log(sigma_sq_mutual), axis=-1)
    dist = dist1 + dist2
    print('   s2', time.time()-t1)
    return -dist, -dist1, -dist2


def aggregate_PFE(x, sigma_sq=None, normalize=True, concatenate=False, T=1.):
    if sigma_sq is None:
        D = int(x.shape[1] / 2)
        if x.shape[1] == 257 or x.shape[1] == 513:
            D = int(x.shape[1] - 1)
        mu, sigma_sq = x[:,:D], x[:,D:]
    else:
        mu = x
    attention = 1. / sigma_sq
    attention = np.exp(np.log(attention+1e-6)/T)
    attention = attention / np.sum(attention, axis=0, keepdims=True)

    mu_new = np.sum(mu * attention, axis=0)
    # sigma_sq_new = np.min(sigma_sq, axis=0)
    sigma_sq_new = np.sum(sigma_sq * attention, axis=0)

    if normalize:
        mu_new = l2_normalize(mu_new)

    if concatenate:
        return np.concatenate([mu_new, sigma_sq_new])
    else:
        return mu_new, sigma_sq_new


class _PartitionIterator(with_metaclass(ABCMeta)):
    """Base class for CV iterators where train_mask = ~test_mask

    Implementations must define `_iter_test_masks` or `_iter_test_indices`.

    Parameters
    ----------
    n : int
        Total number of elements in dataset.
    """

    def __init__(self, n):
        if abs(n - int(n)) >= np.finfo('f').eps:
            raise ValueError("n must be an integer")
        self.n = int(n)

    def __iter__(self):
        ind = np.arange(self.n)
        for test_index in self._iter_test_masks():
            train_index = np.logical_not(test_index)
            train_index = ind[train_index]
            test_index = ind[test_index]
            yield train_index, test_index

    # Since subclasses must implement either _iter_test_masks or
    # _iter_test_indices, neither can be abstract.
    def _iter_test_masks(self):
        """Generates boolean masks corresponding to test sets.

        By default, delegates to _iter_test_indices()
        """
        for test_index in self._iter_test_indices():
            test_mask = self._empty_mask()
            test_mask[test_index] = True
            yield test_mask

    def _iter_test_indices(self):
        """Generates integer indices corresponding to test sets."""
        raise NotImplementedError

    def _empty_mask(self):
        return np.zeros(self.n, dtype=np.bool)


class KFold(with_metaclass(ABCMeta, _PartitionIterator)):
    def __init__(self, n, n_folds=3, shuffle=False,
                 random_state=None):
        super(KFold, self).__init__(n)
        self.n = int(n)
        self.n_folds = n_folds
        self.shuffle = shuffle
        self.random_state = random_state
        self.idxs = np.arange(n)
        if shuffle:
            np.random.shuffle(self.idxs)

    def _iter_test_indices(self):
        n = self.n
        n_folds = self.n_folds
        fold_sizes = (n // n_folds) * np.ones(n_folds, dtype=np.int)
        fold_sizes[:n % n_folds] += 1
        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            yield self.idxs[start:stop]
            current = stop

    def __repr__(self):
        return '%s.%s(n=%i, n_folds=%i, shuffle=%s, random_state=%s)' % (
            self.__class__.__module__,
            self.__class__.__name__,
            self.n,
            self.n_folds,
            self.shuffle,
            self.random_state,
        )

    def __len__(self):
        return self.n_folds

