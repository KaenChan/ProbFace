"""Test PFE on LFW.
"""
# MIT License
# 
# Copyright (c) 2019 Yichun Shi
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

from utils import utils
from utils.imageprocessing import preprocess
from utils.dataset import Dataset
from network import Network
from evaluation.lfw import LFWTest


def main(args):


    paths = Dataset(args.dataset_path)['abspath']
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
    print('mu', np.max(mu), np.min(mu), np.mean(mu))
    print('sigma_sq', np.max(sigma_sq), np.min(sigma_sq), np.mean(sigma_sq))

    quality_score = -np.mean(np.log(sigma_sq), axis=1)
    print('quality_score quality_score=-np.mean(np.log(sigma_sq),axis=1) percentile [0, 10, 30, 50, 70, 90, 100]')
    print('quality_score ', np.percentile(quality_score.ravel(), [0, 10, 30, 50, 70, 90, 100]))

    feat_pfe = np.concatenate([mu, sigma_sq], axis=1)

    lfwtest = LFWTest(paths)
    lfwtest.init_standard_proto(args.protocol_path)

    accuracy, threshold = lfwtest.test_standard_proto(mu, utils.pair_cosin_score)
    print('Euclidean (cosine) accuracy: %.5f threshold: %.5f' % (accuracy, threshold))
    accuracy, threshold = lfwtest.test_standard_proto(feat_pfe, utils.pair_MLS_score)
    print('MLS accuracy: %.5f threshold: %.5f' % (accuracy, threshold))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", help="The path to the pre-trained model directory",
                        type=str,
                        default=r'D:\chenkai\Probabilistic-Face-Embeddings-master\log\resface64_relu_msarcface_am_PFE\20191208-232851-99.817')
    parser.add_argument("--dataset_path", help="The path to the LFW dataset directory",
                        type=str, default=r'F:\data\face-recognition\lfw\lfw-112-mxnet')
    parser.add_argument("--protocol_path", help="The path to the LFW protocol file",
                        type=str, default=r'D:\chenkai\Probabilistic-Face-Embeddings-master\proto\lfw_pairs.txt')
    parser.add_argument("--batch_size", help="Number of images per mini batch",
                        type=int, default=128)
    args = parser.parse_args()
    main(args)
