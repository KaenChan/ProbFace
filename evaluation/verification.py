"""Helper for evaluation on the Labeled Faces in the Wild dataset 
"""

# MIT License
# 
# Copyright (c) 2016 David Sandberg
# Copyright (c) 2016 Kaen Chan
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from utils import utils
from utils.imageprocessing import preprocess
from utils.dataset import Dataset
from network import Network

import os
import argparse
import sys
import numpy as np
from scipy import misc
from scipy import interpolate
import sklearn
import cv2
import math
import datetime
import pickle
from sklearn.decomposition import PCA
import mxnet as mx
from mxnet import ndarray as nd

from utils.utils import KFold


def calculate_roc(embeddings1, embeddings2, actual_issame, compare_func, nrof_folds=10):
    assert(embeddings1.shape[0] == embeddings2.shape[0])
    assert(embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    k_fold = KFold(nrof_pairs, n_folds=nrof_folds, shuffle=False)

    accuracy = np.zeros((nrof_folds))
    indices = np.arange(nrof_pairs)
    #print('pca', pca)

    accuracies = np.zeros(10, dtype=np.float32)
    thresholds = np.zeros(10, dtype=np.float32)

    dist = compare_func(embeddings1, embeddings2)

    for fold_idx, (train_set, test_set) in enumerate(k_fold):
        #print('train_set', train_set)
        #print('test_set', test_set)
        # Find the best threshold for the fold
        from evaluation import metrics
        train_score = dist[train_set]
        train_labels = actual_issame[train_set] == 1
        acc, thresholds[fold_idx] = metrics.accuracy(train_score, train_labels)
        # print('train acc', acc, thresholds[i])

        # Testing
        test_score = dist[test_set]
        accuracies[fold_idx], _ = metrics.accuracy(test_score, actual_issame[test_set]==1, np.array([thresholds[fold_idx]]))

    accuracy = np.mean(accuracies)
    threshold = np.mean(thresholds)
    return accuracy, threshold


def evaluate(embeddings, actual_issame, compare_func, nrof_folds=10):
    # Calculate evaluation metrics
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    actual_issame = np.asarray(actual_issame)
    return calculate_roc(embeddings1, embeddings2,
                         actual_issame, compare_func, nrof_folds=nrof_folds)


def load_bin(path, image_size):
  print(path, image_size)
  with open(path, 'rb') as f:
      if 'lfw_all' in path:
          bins, issame_list = pickle.load(f)
      else:
          bins, issame_list = pickle.load(f, encoding='latin1')
  data_list = []
  for flip in [0]:
    data = nd.empty((len(issame_list)*2, image_size[0], image_size[1], 3))
    data_list.append(data)
  print(len(bins))
  for i in range(len(issame_list)*2):
    _bin = bins[i]
    # print(type(_bin))
    img = mx.image.imdecode(_bin)
    # img = nd.transpose(img, axes=(2, 0, 1))
    for flip in [0]:
      if flip==1:
        img = mx.ndarray.flip(data=img, axis=2)
      data_list[flip][i][:] = img
    if i%1000==0:
      print('loading bin', i)
  print(data_list[0].shape)
  return (data_list, issame_list)


def eval_images(images_preprocessed, issame_list, network, batch_size, nfolds=10, name=''):
    print('testing verification..')
    images = images_preprocessed
    print(images.shape)
    mu, sigma_sq = network.extract_feature(images, batch_size, verbose=True)
    sigma_sq = sigma_sq[..., :1]
    feat_pfe = np.concatenate([mu, sigma_sq], axis=1)

    if name != '':
        np.save('o_sigma_%s.npy' % name, sigma_sq)

    s = 'sigma_sq ' + str(np.percentile(sigma_sq.ravel(), [0, 10, 30, 50, 70, 90, 100])) + ' percentile [0, 10, 30, 50, 70, 90, 100]\n'
    # print(mu.shape)
    accuracy, threshold = evaluate(mu, issame_list, utils.pair_cosin_score, nrof_folds=nfolds)
    s += 'Cosine score acc %f threshold %f\n' % (accuracy, threshold)
    # print('cosin', 'acc', accuracy, 'threshold', threshold)
    print(s)
    compare_func = lambda x,y: utils.pair_MLS_score(x, y, use_attention_only=False)
    accuracy, threshold = evaluate(feat_pfe, issame_list, compare_func, nrof_folds=nfolds)
    s += 'MLS score acc %f threshold %f' % (accuracy, threshold)
    # print('MLS', 'acc', accuracy, 'threshold', threshold)
    compare_func = lambda x,y: utils.pair_MLS_score(x, y, use_attention_only=True)
    accuracy, threshold = evaluate(feat_pfe, issame_list, compare_func, nrof_folds=nfolds)
    s += '\nAttention-only score acc %f threshold %f' % (accuracy, threshold)
    print(s)
    return s


def eval(data_set, network, batch_size, nfolds=10, name=''):
  print('testing verification..')
  data_list = data_set[0]
  issame_list = data_set[1]
  data_list = data_list[0].asnumpy()
  images = preprocess(data_list, network.config, False)
  del data_set
  return eval_images(images, issame_list, network, batch_size, nfolds=10, name=name)


def main(args):
    data_dir = args.dataset_path
    # data_dir = r'F:\data\face-recognition\MS-Celeb-1M\faces_emore'
    # data_dir = r'F:\data\face-recognition\trillion-pairs\challenge\ms1m-retinaface-t1'

    # Load model files and config file
    network = Network()
    network.load_model(args.model_dir)

    for name in args.target.split(','):
        path = os.path.join(data_dir,name+".bin")
        if os.path.exists(path):
            image_size = [112, 112]
            data_set = load_bin(path, image_size)
            print('ver', name)
            info = eval(data_set, network, args.batch_size, 10, name=name)
            # print(info)
            info_result = '--- ' + name + ' ---\n'
            info_result += info + "\n"
            print("")
            print(info_result)
            with open(os.path.join(args.model_dir, 'testing-log.txt'), 'a') as f:
                f.write(info_result + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", help="The path to the pre-trained model directory",
                        type=str,
                        default=r'')
    parser.add_argument("--dataset_path", help="The path to the LFW dataset directory",
                        type=str, default=r'F:\data\face-recognition\trillion-pairs\challenge\ms1m-retinaface-t1')
    parser.add_argument("--batch_size", help="Number of images per mini batch",
                        type=int, default=64)
    parser.add_argument('--target', type=str, default='lfw,cfp_fp,agedb_30', help='verification targets')
    args = parser.parse_args()
    # args.target = 'cfp_fp'
    # args.target = 'agedb_30'
    # args.target = 'cfp_fp,agedb_30'
    # args.target = 'calfw,cplfw,cfp_ff,cfp_fp,agedb_30,vgg2_fp'
    main(args)

