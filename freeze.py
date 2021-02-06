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
    # Load model files and config file
    network = Network()
    args.batch_size = None
    args.model_dir = r'log/resface64/20201028-025034-mls1.0-L1-0.001-triplet0.0001'
    network.freeze_model(args.model_dir)
    # network.config.preprocess_train = []
    # network.config.preprocess_test = []


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", help="The path to the pre-trained model directory",
                        type=str)
    parser.add_argument("--batch_size", help="Number of images per mini batch",
                        type=int, default=128)
    args = parser.parse_args()
    main(args)
