"""Main training file for PFE
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
import time
import imp
import argparse
import tensorflow as tf
import numpy as np

from evaluation.openset_lfw.openset_lfw import get_paths_all, openset_lfw
from utils import utils
from utils.imageprocessing import preprocess
from utils.dataset import Dataset
from network import Network

# from evaluation import verification
from evaluation import verification_risk as verification


def main(args):
    print('start main')
    test_1v1_target = 'cfp_fp,agedb_30'
    test_1v1_target = 'cfp_fp'
    test_lfw_openset_numTrials = 0

    # I/O
    config_file = args.config_file
    config = imp.load_source('config', config_file)
    if args.name:
        config.name = args.name

    t1 = time.time()
    read_imagelist_from_file = False
    imagelist_file_for_train = 'data/list_to_train_ms1m-retinaface-t1-img.txt'
    if read_imagelist_from_file:
        trainset = Dataset(imagelist_file_for_train)
        print('time', time.time() - t1)
    else:
        trainset = Dataset(config.train_dataset_path)
        print('time', time.time() - t1)
        # trainset.write_datalist_to_file(imagelist_file_for_train)
    trainset.set_base_seed(config.base_random_seed)

    network = Network()
    network.initialize(config, trainset.num_classes)

    # Initalization for running
    log_dir = utils.create_log_dir(config, config_file)
    summary_writer = tf.summary.FileWriter(log_dir, network.graph)
    if config.restore_model:
        print(config.restore_model)
        network.restore_model(config.restore_model, config.restore_scopes, config.exclude_restore_scopes)

    test_images_lfw = None
    if test_lfw_openset_numTrials > 0 and args.dataset_path:
        lfw_paths = get_paths_all(os.path.expanduser(args.dataset_path))
        test_images_lfw = preprocess(lfw_paths, config, False)

    ver_list = []
    ver_name_list = []
    for name in test_1v1_target.split(','):
        path = os.path.join(config.test_data_dir_mx,name+".bin")
        if os.path.exists(path):
            image_size = [112, 112]
            data_list, issame_list = verification.load_bin(path, image_size)
            data_list = data_list[0].asnumpy()
            images = preprocess(data_list, network.config, False)
            data_set = (images, issame_list)
            ver_list.append(data_set)
            ver_name_list.append(name)
            print('ver', name)

    proc_func = lambda images: preprocess(images, config, True)
    trainset.start_batch_queue(config.batch_format, proc_func=proc_func)
    # batch = trainset.pop_batch_queue()

    # Main Loop
    print('\nStart Training\nname: {}\n# epochs: {}\nepoch_size: {}\nbatch_size: {}\n'.format(
            config.name, config.num_epochs, config.epoch_size, config.batch_format['size']))
    global_step = 0
    network.save_model(log_dir, global_step)
    start_time = time.time()
    for epoch in range(config.num_epochs+1):

        # Save the model
        network.save_model(log_dir, global_step)

        if epoch > 0:
            info_w = ''
            if test_lfw_openset_numTrials > 0 and args.dataset_path:
                mu, sigma_sq = network.extract_feature(test_images_lfw, 64, verbose=True)
                quality_score = -np.mean(np.log(sigma_sq), axis=1)
                print('sigma_sq percentile [0, 10, 30, 50, 70, 90, 100]')
                print('sigma_sq ', np.percentile(quality_score.ravel(), [0, 10, 30, 50, 70, 90, 100]))
                feat_pfe = np.concatenate([mu, sigma_sq], axis=1)
                info1 = openset_lfw(mu, utils.pair_cosin_score, test_lfw_openset_numTrials)
                info_w += info1 + '\n'
                print(info1)
                info2 = openset_lfw(feat_pfe, utils.nvm_MLS_score, test_lfw_openset_numTrials)
                print(info2)
                info_w += info2 + '\n'
                info3 = openset_lfw(feat_pfe, utils.nvm_MLS_score_attention, test_lfw_openset_numTrials)
                print(info3)
                info_w += info3 + '\n'
            info_ver = ''
            for i in range(len(ver_list)):
                print('---', ver_name_list[i], '---')
                info_ver_ = verification.eval_images(ver_list[i][0], ver_list[i][1], network, 128, 10)
                print(info_ver_)
                info_ver += '---' + ver_name_list[i] + '\n'
                info_ver += info_ver_ + '\n'
            info_w += info_ver + '\n'
            with open(os.path.join(log_dir, 'training-log.txt'), 'a') as f:
                f.write(info_w)
        if epoch == config.num_epochs:
            break

        # Training
        for step in range(config.epoch_size):
            # Prepare input
            learning_rate = utils.get_updated_learning_rate(global_step, config)
            batch = trainset.pop_batch_queue()
            if len(batch['image']) > len(batch['label']):
                batch['label'] = np.concatenate([batch['label'], batch['label']], axis=0)

            wl, global_step = network.train(batch['image'], batch['label'], learning_rate, config.keep_prob)

            wl['lr'] = learning_rate

            # Display
            if step % config.summary_interval == 0:
                duration = time.time() - start_time
                start_time = time.time()
                with open(os.path.join(log_dir, 'training-log.txt'), 'a') as f:
                    s = utils.display_info(epoch, step, duration, wl)
                    print(s)
                    f.write(s + '\n')


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", help="The path to the training configuration file",
                        type=str)
    parser.add_argument("--name", help="Rename the log dir",
                        type=str, default=None)
    parser.add_argument("--dataset_path", help="The path to the LFW dataset directory",
                        type=str, default=r'F:\data\face-recognition\lfw\lfw-112-mxnet')
    args = parser.parse_args()
    main(args)
