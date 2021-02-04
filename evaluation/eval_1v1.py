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


    paths = [
        r'F:\data\face-recognition\lfw\lfw-112-mxnet\Abdoulaye_Wade\Abdoulaye_Wade_0002.jpg',
        r'F:\data\face-recognition\lfw\lfw-112-mxnet\Abdoulaye_Wade\Abdoulaye_Wade_0003.jpg',
        r'F:\data\face-recognition\realsense\data-labeled-clean-strict2-112-mxnet\rgb\001-chenkai\a-000013.jpg',
        r'F:\data\face-recognition\realsense\data-labeled-clean-strict2-112-mxnet\rgb\001-chenkai\rgb_2.jpg',
        r'F:\data\face-recognition\lfw\lfw-112-mxnet\Abdoulaye_Wade\Abdoulaye_Wade_0002.jpg',
        r'F:\data\face-recognition\realsense\data-labeled-clean-strict2-112-mxnet\rgb\001-chenkai\rgb_2.jpg',
        r'F:\data\face-recognition\lfw\lfw-112-mxnet\Abdoulaye_Wade\Abdoulaye_Wade_0003.jpg',
        r'F:\data\face-recognition\realsense\data-labeled-clean-strict2-112-mxnet\rgb\001-chenkai\rgb_2.jpg',
    ]
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
    print(mu.shape, sigma_sq.shape)

    print('sigma_sq', np.max(sigma_sq), np.min(sigma_sq), np.mean(sigma_sq), np.exp(np.mean(np.log(sigma_sq))))
    log_sigma_sq = np.log(sigma_sq)
    print('log_sigma_sq', np.max(log_sigma_sq), np.min(log_sigma_sq), np.mean(log_sigma_sq))
    # print('sigma_sq', sigma_sq)

    feat_pfe = np.concatenate([mu, sigma_sq], axis=1)

    score = utils.pair_cosin_score(mu[::2], mu[1::2])
    print(score)

    score = utils.pair_MLS_score(feat_pfe[::2], feat_pfe[1::2])
    print(score)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", help="The path to the pre-trained model directory",
                        type=str,
                        default=r'D:\chenkai\Probabilistic-Face-Embeddings-master\log\resface64_relu_msarcface_am_PFE/20191229-172304-iter15')
    parser.add_argument("--batch_size", help="Number of images per mini batch",
                        type=int, default=128)
    args = parser.parse_args()
    main(args)
