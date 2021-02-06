"""Test PFE on LFW.
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

import os
import sys
import imp
import argparse
import time
import math
import numpy as np

from evaluation.openset_lfw.openset_lfw import get_paths_all, openset_lfw
from utils import utils
from utils.imageprocessing import preprocess
from utils.dataset import Dataset
from network import Network
from evaluation.lfw import LFWTest


def main(args):

    paths = get_paths_all(os.path.expanduser(args.dataset_path))
    print('%d images to load.' % len(paths))
    assert(len(paths)>0)

    # Load model files and config file
    network = Network()
    network.load_model(args.model_dir)
    # network.config.preprocess_train = []
    # network.config.preprocess_test = []
    images = preprocess(paths, network.config, False)
    import cv2
    # images = np.array([cv2.resize(img, (96, 96)) for img in images])
    # images = (images - 128.) / 128.
    # images = images[..., ::-1]
    print(images.shape)
    # print(images[0,:5,:5,0])

    # Run forward pass to calculate embeddings
    mu, sigma_sq = network.extract_feature(images, args.batch_size, verbose=True)
    sigma_sq = sigma_sq[..., :1]
    feat_pfe = np.concatenate([mu, sigma_sq], axis=1)

    quality_score = -np.mean(np.log(sigma_sq), axis=1)
    print('quality_score quality_score=-np.mean(np.log(sigma_sq),axis=1) percentile [0, 10, 30, 50, 70, 90, 100]')
    print('quality_score ', np.percentile(quality_score.ravel(), [0, 10, 30, 50, 70, 90, 100]))
    print('quality_score sigma_sq ', np.percentile(np.mean(sigma_sq, axis=1), [0, 10, 30, 50, 70, 90, 100]))

    lfwtest = LFWTest(paths)
    lfwtest.init_standard_proto(args.protocol_path)

    numTrials = 1
    numTrials = 10
    info1 = openset_lfw(mu, utils.pair_cosin_score, numTrials)
    print(info1)

    compare_func = lambda x,y: utils.nvm_MLS_score(x, y)
    info2 = openset_lfw(feat_pfe, compare_func, numTrials)
    print(info2)
    # compare_func = lambda x,y: utils.nvm_MLS_score_attention(x, y)
    # info3 = openset_lfw(feat_pfe, compare_func, numTrials)
    # print(info3)
    print('-----------')
    print(info1)
    print(info2)
    # print(info3)
    with open(os.path.join(args.model_dir, 'testing-log.txt'), 'a') as f:
        f.write(info1 + '\n')
        f.write(info2 + '\n')
        # f.write(info3 + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", help="The path to the pre-trained model directory",
                        type=str)
    parser.add_argument("--dataset_path", help="The path to the LFW dataset directory",
                        type=str, default='data/lfw_mtcnncaffe_aligned')
    parser.add_argument("--protocol_path", help="The path to the LFW protocol file",
                        type=str, default='./proto/lfw_pairs.txt')
    parser.add_argument("--batch_size", help="Number of images per mini batch",
                        type=int, default=128)
    args = parser.parse_args()
    main(args)
