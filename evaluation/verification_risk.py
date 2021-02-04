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
import _pickle as cPickle

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


def evaluate(embeddings, actual_issame, compare_func, nrof_folds=10, keep_idxes=None):
    # Calculate evaluation metrics
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    actual_issame = np.asarray(actual_issame)
    if keep_idxes is not None:
        embeddings1 = embeddings1[keep_idxes]
        embeddings2 = embeddings2[keep_idxes]
        actual_issame = actual_issame[keep_idxes]
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


def eval_images_cmp(images_preprocessed, issame_list, network, batch_size, nfolds=10, name='', result_dir='',
                re_extract_feature=False, filter_out_type='max'):
    print('testing verification..')
    result_dir = r'G:\chenkai\Probabilistic-Face-Embeddings-master\log\resface64_relu_msarcface_am_PFE\20201020-180059-mls-only'
    save_name_pkl_feature = result_dir + '/%s_feature.pkl' % name
    if re_extract_feature or not os.path.exists(save_name_pkl_feature):
        images = images_preprocessed
        print(images.shape)
        mu, sigma_sq = network.extract_feature(images, batch_size, verbose=True)
        save_data = (mu, sigma_sq)
        with open(save_name_pkl_feature, 'wb') as f:
            cPickle.dump(save_data, f)
        print('save', save_name_pkl_feature)
    else:
        with open(save_name_pkl_feature, 'rb') as f:
            data = cPickle.load(f)
        mu, sigma_sq = data
        print('load', save_name_pkl_feature)
    feat_pfe = np.concatenate([mu, sigma_sq], axis=1)
    threshold = -1.604915

    result_dir2 = r'G:\chenkai\Probabilistic-Face-Embeddings-master\log\resface64m_relu_msarcface_am_PFE\20201028-193142-mls1.0-abs0.001-triplet0.0001'
    save_name_pkl_feature2 = result_dir2 + '/%s_feature.pkl' % name
    with open(save_name_pkl_feature2, 'rb') as f:
        data = cPickle.load(f)
    mu2, sigma_sq2 = data
    print('load', save_name_pkl_feature2)
    feat_pfe2 = np.concatenate([mu2, sigma_sq2], axis=1)
    threshold2 = -1.704115

    embeddings = mu
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    cosine_score = np.sum(embeddings1 * embeddings2, axis=1)

    compare_func = lambda x,y: utils.pair_MLS_score(x, y, q=0, use_attention_only=False)
    embeddings = feat_pfe
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    actual_issame = np.asarray(issame_list)
    dist = compare_func(embeddings1, embeddings2)
    pred = (dist > threshold).astype(int)
    idx_true1 = pred == actual_issame
    print('acc1', np.average(idx_true1), np.sum(idx_true1))

    embeddings = feat_pfe2
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    actual_issame = np.asarray(issame_list)
    dist = compare_func(embeddings1, embeddings2)
    pred = (dist > threshold2).astype(int)
    idx_true2 = pred == actual_issame
    print('acc2', np.average(idx_true2), np.sum(idx_true2))

    idx_keep = np.where((~idx_true1) * (idx_true2) == 1)[0]

    print('idx_keep', len(idx_keep))

    for idx in idx_keep:
        i, j = idx*2, idx*2 + 1
        print(idx, i, j, 'label', issame_list[idx], 'cos', cosine_score[idx], end=' ')
        print('sigma1', sigma_sq[i], sigma_sq[j], end=' ')
        print('sigma2', sigma_sq2[i], sigma_sq2[j])
        data_dir = r'F:\data\face-recognition\test\1v1\cplfw'
        name_jpg = '%04d_%d.jpg' % (i, issame_list[idx])
        path1 = os.path.join(data_dir, name_jpg)
        path2 = os.path.join(data_dir + '_sel', name_jpg)
        import shutil
        shutil.copy(path1, path2)
        name_jpg = '%04d_%d.jpg' % (j, issame_list[idx])
        path1 = os.path.join(data_dir, name_jpg)
        path2 = os.path.join(data_dir + '_sel', name_jpg)
        import shutil
        shutil.copy(path1, path2)
    exit(0)


def extract_features(images_preprocessed, issame_list, network, batch_size, name='', result_dir='',
                re_extract_feature=True):
    print('testing verification..')
    save_name_pkl_feature = result_dir + '/%s_feature.pkl' % name
    if re_extract_feature or not os.path.exists(save_name_pkl_feature):
        images = images_preprocessed
        print(images.shape)
        mu, sigma_sq = network.extract_feature(images, batch_size, verbose=True)
        save_data = (mu, sigma_sq, issame_list)
        with open(save_name_pkl_feature, 'wb') as f:
            cPickle.dump(save_data, f)
        print('save', save_name_pkl_feature)
    else:
        with open(save_name_pkl_feature, 'rb') as f:
            data = cPickle.load(f)
        if len(data) == 3:
            mu, sigma_sq, issame_list = data
        else:
            mu, sigma_sq = data
        print('load', save_name_pkl_feature)
    return mu, sigma_sq, issame_list

def eval_images_with_sigma(mu, sigma_sq, issame_list, nfolds=10, name='', filter_out_type='max'):
    print('sigma_sq', sigma_sq.shape)
    feat_pfe = np.concatenate([mu, sigma_sq], axis=1)

    #2
    if False:
        # result_dir = r'G:\chenkai\Probabilistic-Face-Embeddings-master\log\resface64_relu_msarcface_am_PFE\20201020-180059-mls-only'
        result_dir = r'G:\chenkai\Probabilistic-Face-Embeddings-master\log\resface64_128x128_dropout\20201114-171043-iter0'
        result_dir = r'G:\chenkai\Probabilistic-Face-Embeddings-master\log\resface64m_relu_msarcface_am_PFE\20201028-193142-mls1.0-abs0.001-triplet0.0001'
        result_dir = r'G:\chenkai\Probabilistic-Face-Embeddings-master\log\resface64_relu_msarcface_am_PFE_res12\20201031-181621-mls1.0-abs0.001'
        result_dir = r'G:\chenkai\Probabilistic-Face-Embeddings-master\log\resface64s2\20201030-125503-ms'
        result_dir = r'G:\chenkai\Probabilistic-Face-Embeddings-master\log\resface64_relu_msarcface_am_PFE_mbv3\20201113-000349-run02-iter64-99.03'
        result_dir = r'G:\chenkai\Probabilistic-Face-Embeddings-master\log\resface64_relu_msarcface_am_PFE_mbv3\20201113-151004-multiscale-abs1e-3-triplet1e-3'
        save_name_pkl_feature = result_dir + '/%s_feature.pkl' % 'cfp_fp'
        with open(save_name_pkl_feature, 'rb') as f:
            data = cPickle.load(f)
        if len(data) == 3:
            mu2, sigma_sq2, issame_list2 = data
        else:
            mu2, sigma_sq2 = data
        print('load', save_name_pkl_feature)
        print('sigma_sq2', sigma_sq2.shape)
        # sigma_sq = np.concatenate([sigma_sq, sigma_sq2], axis=-1)
        sigma_sq = sigma_sq2

    if False:
        # result_dir = r'G:\chenkai\Probabilistic-Face-Embeddings-master\log\resface64_relu_msarcface_am_PFE\20201020-180059-mls-only'
        # result_dir = r'G:\chenkai\Probabilistic-Face-Embeddings-master\log\resface64s2\20201030-125503-ms'
        result_dir = r'G:\chenkai\Probabilistic-Face-Embeddings-master\log\resface64m_relu_msarcface_am_PFE\20201114-181703-dropout'
        result_dir = r'G:\chenkai\Probabilistic-Face-Embeddings-master\log\resface64_128x128_dropout\20201114-171043-iter0'
        mu_list = []
        sigma_list = []
        file_list = os.listdir(result_dir)
        file_list = [l for l in file_list if 'pkl' in l]
        file_list = [l for l in file_list if 'keep0.9' in l]
        print('file_list', len(file_list))
        for l in file_list[:25]:
            save_name_pkl_feature = result_dir + '/%s' % l
            # print(save_name_pkl_feature)
            with open(save_name_pkl_feature, 'rb') as f:
                data = cPickle.load(f)
            mu2, sigma_sq2 = data
            mu_list += [mu2]
            sigma_list += [sigma_sq2]
        mu_list = np.array(mu_list)  # 71x10000x256
        sigma_list = np.array(sigma_list)
        mu_mean = np.mean(mu_list, axis=0) # 10000x256
        mu_mean = mu_mean / np.linalg.norm(mu_mean, axis=-1, keepdims=True)
        score1 = np.sum(mu * mu_mean, axis=-1)
        print(score1[:5])
        if True:
            mu_mean = np.array([mu_mean])
            score = np.sum(mu_list * mu_mean, axis=-1)
            score_var = np.std(score, axis=0)
            sigma_sq = score_var.reshape([-1, 1])
        if False:
            dist_list = []
            for i in range(len(mu_list)):
                for j in range(i+1, len(mu_list)):
                    dist = 1 - np.sum(mu_list[i] * mu_list[j], axis=-1)
                    dist_list += [dist]
            score_var = np.mean(dist_list, axis=0)
            sigma_sq = score_var.reshape([-1, 1])

    if name != '':
        np.save('o_sigma_%s.npy' % name, sigma_sq)

    # quality_score = -np.mean(np.log(sigma_sq), axis=1)
    # print('quality_score quality_score=-np.mean(np.log(sigma_sq),axis=1) percentile [0, 10, 30, 50, 70, 90, 100]')
    # print('quality_score ', np.percentile(quality_score.ravel(), [0, 10, 30, 50, 70, 90, 100]))

    s = 'sigma_sq ' + str(np.percentile(sigma_sq.ravel(), [0, 10, 30, 50, 70, 90, 100])) + ' percentile [0, 10, 30, 50, 70, 90, 100]\n'
    # print(mu.shape)

    print('sigma_sq', sigma_sq.shape)
    if sigma_sq.shape[1] == 2:
        sigma_sq_c = np.copy(sigma_sq)
        sigma_sq_list = [sigma_sq_c[:,:1], sigma_sq_c[:,1:]]
    else:
        sigma_sq_list = [sigma_sq]
    for sigma_sq in sigma_sq_list:
        sigma_sq1 = sigma_sq[0::2]
        sigma_sq2 = sigma_sq[1::2]
        # filter_out_type = 'max'
        if filter_out_type == 'max':
            sigma_fuse = np.maximum(sigma_sq1, sigma_sq2)
        else:
            sigma_fuse = sigma_sq1 + sigma_sq2
        # risk_factor = 0.1
        for risk_factor in [0.0, 0.1, 0.2, 0.3]:
            risk_threshold = np.percentile(sigma_fuse.ravel(), (1-risk_factor)*100)
            keep_idxes = np.where(sigma_fuse <= risk_threshold)[0]
            s += 'risk_factor {} '.format(risk_factor)
            s += 'risk_threshold {} '.format(risk_threshold)
            s += 'keep_idxes {} / {} '.format(len(keep_idxes), len(sigma_fuse))
            if len(keep_idxes) == 0:
                keep_idxes = None

            accuracy, threshold = evaluate(mu, issame_list, utils.pair_cosin_score, nrof_folds=nfolds, keep_idxes=keep_idxes)
            s += 'Cosine score acc %f threshold %f\n' % (accuracy, threshold)
            # print('cosin', 'acc', accuracy, 'threshold', threshold)
            # print(s)
            # for q in [0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6]:
            for q in [0]:
                compare_func = lambda x,y: utils.pair_MLS_score(x, y, q=q, use_attention_only=False)
                # accuracy, threshold = evaluate(feat_pfe, issame_list, compare_func, nrof_folds=nfolds, keep_idxes=keep_idxes)
                # # s += '\n========> q' + str(q) +' '
                # s += 'MLS score acc %f threshold %f' % (accuracy, threshold)
                # print('MLS', 'acc', accuracy, 'threshold', threshold)
                # compare_func = lambda x,y: utils.pair_MLS_score(x, y, q=q, use_attention_only=True)
                # accuracy, threshold = evaluate(feat_pfe, issame_list, compare_func, nrof_folds=nfolds, keep_idxes=keep_idxes)
                # # s += '\n========> q' + str(q) +' '
                # s += '\nAttention-only score acc %f threshold %f' % (accuracy, threshold)
            if keep_idxes is None:
                break
    # print(s)
    return s


def eval_images(images_preprocessed, issame_list, network, batch_size, nfolds=10, name='', result_dir='',
                re_extract_feature=True, filter_out_type='max'):
    mu, sigma_sq, issame_list = extract_features(images_preprocessed, issame_list, network, batch_size,
                                                 name=name, result_dir=result_dir,
                                                 re_extract_feature=re_extract_feature)
    return eval_images_with_sigma(mu, sigma_sq, issame_list, nfolds=10, name='', filter_out_type=filter_out_type)


def save_dataset_as_jpg(data_set, name):
    data_list = data_set[0]
    issame_list = data_set[1]
    data_list = data_list[0].asnumpy()
    root = r'F:\data\face-recognition\test\1v1'
    for i in range(len(data_list)):
        path = os.path.join(root, name)
        if not os.path.exists(path):
            os.makedirs(path)
        path = os.path.join(path, '%04d_%d.jpg' % (i, issame_list[i//2]))
        print(path)
        cv2.imwrite(path, data_list[i].astype(np.uint8)[...,::-1])


def eval(data_set, network, batch_size, nfolds=10, name='', result_dir='', re_extract_feature=True, filter_out_type='max'):
  print('testing verification..')
  data_list = data_set[0]
  issame_list = data_set[1]
  data_list = data_list[0].asnumpy()
  images = preprocess(data_list, network.config, False)
  del data_set
  for i in range(1):
      # name1 = name + '_keep0.9_%03d' % i
      name1 = name
      ret = eval_images(images, issame_list, network, batch_size, nfolds=10, name=name1, result_dir=result_dir,
                         re_extract_feature=re_extract_feature, filter_out_type=filter_out_type)
      print(ret)
      # ret = eval_images_cmp(images, issame_list, network, batch_size, nfolds=10, name=name, result_dir=result_dir,
      #                       re_extract_feature=re_extract_feature, filter_out_type=filter_out_type)
  return ret


def main(args):
    data_dir = args.dataset_path
    # data_dir = r'F:\data\face-recognition\MS-Celeb-1M\faces_emore'
    # data_dir = r'F:\data\face-recognition\trillion-pairs\challenge\ms1m-retinaface-t1'

    re_extract_feature = True
    # filter_out_type = 'add'
    filter_out_type = 'max'

    # Load model files and config file
    network = Network()
    network.load_model(args.model_dir)

    # # images = np.random.random([1, 128, 128, 3])
    # images = np.random.random([1, 96, 96, 3])
    # for _ in range(5):
    #     mu, sigma_sq = network.extract_feature(images, 1, verbose=True)
    #     print(mu[0, :5])
    # exit(0)

    for namec in args.target.split(','):
        path = os.path.join(data_dir,namec+".bin")
        if os.path.exists(path):
            image_size = [112, 112]
            data_set = load_bin(path, image_size)
            name = namec
            print('ver', name)
            info = eval(data_set, network, args.batch_size, 10, name=name, result_dir=args.model_dir,
                        re_extract_feature=re_extract_feature, filter_out_type=filter_out_type)
            # print(info)
            info_result = '--- ' + name + ' ---\n'
            info_result += info + "\n"
            print("")
            print(info_result)
            with open(os.path.join(args.model_dir, 'testing-log-risk-{}-{}.txt'.format(name, filter_out_type)), 'a') as f:
                f.write(info_result + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", help="The path to the pre-trained model directory",
                        type=str,
                        default=r'')
    parser.add_argument("--dataset_path", help="The path to the LFW dataset directory",
                        type=str, default=r'F:\data\face-recognition\trillion-pairs\challenge\ms1m-retinaface-t1')
    parser.add_argument("--batch_size", help="Number of images per mini batch",
                        type=int, default=16)
    parser.add_argument('--target', type=str, default='lfw,cfp_fp,agedb_30', help='verification targets')
    args = parser.parse_args()
    # args.model_dir = r'log/resface64/20201028-193142-multiscale'
    # args.target = 'lfw,calfw,cplfw,cfp_ff,cfp_fp,agedb_30,vgg2_fp'
    # args.target = 'cfp_fp'
    main(args)
