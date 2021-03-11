"""Main implementation class of PFE
"""
# MIT License
# 
# Copyright (c) 2019 Yichun Shi
# Copyright (c) 2020 Kaen Chan
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
import time

import numpy as np
import tensorflow as tf

from utils.tflib import mutual_likelihood_score_loss


class Network:
    def __init__(self):
        self.graph = tf.Graph()
        gpu_options = tf.GPUOptions(allow_growth=True)
        tf_config = tf.ConfigProto(gpu_options=gpu_options,
                allow_soft_placement=True, log_device_placement=False)
        self.sess = tf.Session(graph=self.graph, config=tf_config)
            
    def initialize(self, config, num_classes=None, for_freeze=False):
        '''
            Initialize the graph from scratch according to config.
        '''
        self.config = config
        if 'loss_weights' not in dir(config):
            self.weight_mls_loss = 5.0
            self.weight_discriminate_loss = 0.0001
            self.weight_output_constraint_loss = 0.001
        else:
            self.weight_mls_loss = config.loss_weights['mls_loss']
            self.weight_discriminate_loss = config.loss_weights['discriminate_loss']
            self.weight_output_constraint_loss = config.loss_weights['output_constraint_loss']

        # discriminat_metric_type = 'contrastive'
        discriminat_metric_type = 'triplet_semihard'
        if 'discriminat_metric_type' not in dir(config):
            discriminat_metric_type = config.discriminat_metric_type

        self.use_mls_loss = self.weight_mls_loss > 0
        self.use_discriminate_loss = self.weight_discriminate_loss > 0

        with self.graph.as_default():
            with self.sess.as_default():
                # Set up placeholders
                h, w = config.image_size
                channels = config.channels

                if for_freeze:
                    self.images = tf.placeholder(tf.float32, shape=[1, h, w, channels], name='input')
                    self.labels = None
                    self.phase_train = False
                    self.learning_rate = None
                    self.keep_prob = 1.
                    self.global_step = 0
                else:
                    self.images = tf.placeholder(tf.float32, shape=[None, h, w, channels], name='images')
                    self.labels = tf.placeholder(tf.int32, shape=[None], name='labels')
                    self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')
                    self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
                    self.phase_train = tf.placeholder(tf.bool, name='phase_train')
                    self.global_step = tf.Variable(0, trainable=False, dtype=tf.int32, name='global_step')

                # Initialialize the backbone network
                network = imp.load_source('embedding_network', config.embedding_network)
                # mu, conv_final = network.inference(self.images, config.embedding_size, keep_probability=self.keep_prob, phase_train=self.phase_train)
                mu, conv_final = network.inference(self.images, config.embedding_size)

                # Initialize the uncertainty module
                uncertainty_module = imp.load_source('uncertainty_module', config.uncertainty_module)
                if 'uncertainty_module_input' not in dir(config):
                    uncertainty_module_input = conv_final
                elif config.uncertainty_module_input == "images":
                    uncertainty_module_input = self.images
                else:
                    uncertainty_module_input = conv_final

                if 'uncertainty_module_output_size' in dir(config):
                    uncertainty_module_output_size = config.uncertainty_module_output_size
                else:
                    uncertainty_module_output_size = config.embedding_size
                    uncertainty_module_output_size = 1
                print('uncertainty_module_output_size', uncertainty_module_output_size)
                scoepname = 'UncertaintyModule'
                log_sigma_sq = uncertainty_module.inference(uncertainty_module_input, uncertainty_module_output_size,
                                        phase_train = self.phase_train, weight_decay = config.weight_decay,
                                        scope=scoepname)

                self.mu = tf.identity(mu, name='mu')
                self.sigma_sq = tf.identity(tf.exp(log_sigma_sq), name='sigma_sq')
                sigma_sq = self.sigma_sq

                if for_freeze:
                    res = tf.concat(values=(self.mu, self.sigma_sq), axis=1, name='embeddings_with_sigma')
                    return

                # Build all losses
                loss_list = []
                self.watch_list = {}

                MLS_loss, attention_loss, attention_pos, attention_neg = mutual_likelihood_score_loss(
                    self.labels, mu, log_sigma_sq, discriminat_metric_type, uncertainty_module_output_size)
                if self.use_mls_loss:
                    loss_list.append(self.weight_mls_loss*MLS_loss)
                    self.watch_list['mls'] = MLS_loss
                self.watch_list['att_pos'] = attention_pos
                self.watch_list['att_neg'] = attention_neg

                output_constraint_type = ''
                if 'output_constraint_type' in dir(config):
                    output_constraint_type = config.output_constraint_type
                if output_constraint_type == 'L2':
                    m0 = tf.stop_gradient(tf.reduce_mean(sigma_sq))
                    s_wd = (sigma_sq/m0 - 1)**2
                    s_wd = tf.reduce_sum(s_wd) * self.weight_output_constraint_loss
                    loss_list.append(s_wd)
                    self.watch_list['s_L2'] = s_wd
                elif output_constraint_type == 'L1':
                    m0 = tf.stop_gradient(tf.reduce_mean(sigma_sq))
                    s_wd = tf.abs(sigma_sq/m0 - 1)
                    s_wd = tf.reduce_sum(s_wd) * self.weight_output_constraint_loss
                    loss_list.append(s_wd)
                    self.watch_list['s_L1'] = s_wd

                if self.use_discriminate_loss:
                    loss_list.append(self.weight_discriminate_loss*attention_loss)
                    self.watch_list['atte'] = attention_loss

                self.watch_list['s_max'] = tf.reduce_max(sigma_sq)
                self.watch_list['s_min'] = tf.reduce_min(sigma_sq)

                # Collect all losses
                reg_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES), name='reg_loss')
                loss_list.append(reg_loss)
                self.watch_list['regu'] = reg_loss

                total_loss = tf.add_n(loss_list, name='total_loss')
                self.watch_list['a'] = total_loss
                grads = tf.gradients(total_loss, self.trainable_variables)

                # Training Operaters
                train_ops = []

                opt = tf.train.MomentumOptimizer(self.learning_rate, momentum=0.9)
                apply_gradient_op = opt.apply_gradients(list(zip(grads, self.trainable_variables)))

                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                train_ops.extend([apply_gradient_op] + update_ops)

                train_ops.append(tf.assign_add(self.global_step, 1))
                self.train_op = tf.group(*train_ops)

                # Initialize variables
                self.sess.run(tf.local_variables_initializer())
                self.sess.run(tf.global_variables_initializer())
                self.saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=1)
 
        return

    @property
    def trainable_variables(self):
        trainable_scopes = 'UncertaintyModule'
        trainable_scopes = trainable_scopes.split(',')
        variables_to_train = []
        for scope in trainable_scopes:
            variables_to_train += [k for k in tf.global_variables() if k.op.name.startswith(scope)]
        print('variables_to_train', len(variables_to_train))
        return variables_to_train

    def save_model(self, model_dir, global_step):
        with self.sess.graph.as_default():
            checkpoint_path = os.path.join(model_dir, 'ckpt')
            metagraph_path = os.path.join(model_dir, 'graph.meta')

            print('Saving variables...')
            self.saver.save(self.sess, checkpoint_path, global_step=global_step, write_meta_graph=False)
            if not os.path.exists(metagraph_path):
                print('Saving metagraph...')
                self.saver.export_meta_graph(metagraph_path)

    def get_model_filenames(self, model_dir):
        files = os.listdir(model_dir)
        meta_files = [s for s in files if s.endswith('.meta')]
        if len(meta_files) == 0:
            raise ValueError('No meta file found in the model directory (%s)' % model_dir)
        elif len(meta_files) > 1:
            raise ValueError('There should not be more than one meta file in the model directory (%s)' % model_dir)
        meta_file = meta_files[0]
        meta_files = [s for s in files if '.ckpt' in s]
        max_step = -1
        ckpt_file = None
        import re
        for f in files:
            step_str = re.match(r'(^model-[\w\- ]+.ckpt-(\d+))', f)
            if step_str is not None and len(step_str.groups()) >= 2:
                step = int(step_str.groups()[1])
                if step > max_step:
                    max_step = step
                    ckpt_file = step_str.groups()[0]
        if ckpt_file is None:
            for f in files:
                if 'index' in f:
                    ckpt_file = f[:-len('.index')]
        meta_file = os.path.join(model_dir, meta_file)
        ckpt_file = os.path.join(model_dir, ckpt_file)
        return meta_file, ckpt_file

    def restore_model(self, model_dir, restore_scopes=None, exclude_restore_scopes=None):
        with self.sess.graph.as_default():
            # var_list = self.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            var_list = tf.trainable_variables()
            print(restore_scopes)
            if exclude_restore_scopes is not None:
                for exclude_restore_scope in exclude_restore_scopes:
                    var_list = [var for var in var_list  if exclude_restore_scope not in var.op.name]
            print(len(var_list))
            meta_file, ckpt_file = self.get_model_filenames(model_dir)

            print('Restoring {} variables from {} ...'.format(len(var_list), ckpt_file))
            saver = tf.train.Saver(var_list)
            saver.restore(self.sess, ckpt_file)

    def load_model(self, model_path, scope=None):
        with self.sess.graph.as_default():
            model_path = os.path.expanduser(model_path)

            # Load grapha and variables separatedly.
            meta_files = [file for file in os.listdir(model_path) if file.endswith('.meta')]
            assert len(meta_files) == 1
            meta_file = os.path.join(model_path, meta_files[0])
            ckpt_file = tf.train.latest_checkpoint(model_path)
            
            print('Metagraph file: %s' % meta_file)
            print('Checkpoint file: %s' % ckpt_file)
            saver = tf.train.import_meta_graph(meta_file, clear_devices=True, import_scope=scope)
            saver.restore(self.sess, ckpt_file)

            # Setup the I/O Tensors
            try:
                self.images = self.graph.get_tensor_by_name('images:0')
                self.mu = self.graph.get_tensor_by_name('mu:0')
                self.sigma_sq = self.graph.get_tensor_by_name('sigma_sq:0')
            except:
                self.images = self.graph.get_tensor_by_name('input:0')
                self.mu = self.graph.get_tensor_by_name('embeddings:0')
                self.sigma_sq = self.graph.get_tensor_by_name('sigma_sq:0')

            self.phase_train = self.graph.get_tensor_by_name('phase_train:0')
            try:
                self.keep_prob = self.graph.get_tensor_by_name('keep_prob:0')
            except:
                print('no keep_prob in load model')
                exit(0)
                self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            self.config = imp.load_source('network_config', os.path.join(model_path, 'config.py'))
            print('mu', self.mu.get_shape())
            print('sigma_sq', self.sigma_sq.get_shape())

    def train(self, images_batch, labels_batch, learning_rate, keep_prob):
        feed_dict = {   self.images: images_batch,
                        self.labels: labels_batch,
                        self.learning_rate: learning_rate,
                        self.keep_prob: keep_prob,
                        self.phase_train: True,}
        _, wl = self.sess.run([self.train_op, self.watch_list], feed_dict = feed_dict)

        step = self.sess.run(self.global_step)

        return wl, step

    def extract_feature(self, images, batch_size, proc_func=None, verbose=False):
        num_images = len(images)
        num_features = self.mu.shape[1]
        mu = np.ndarray((num_images, num_features), dtype=np.float32)
        num_features_sq = self.sigma_sq.shape[1]
        sigma_sq = np.ndarray((num_images, num_features_sq), dtype=np.float32)
        start_time = time.time()
        for start_idx in range(0, num_images, batch_size):
            if verbose:
                elapsed_time = time.strftime('%H:%M:%S', time.gmtime(time.time()-start_time))
                sys.stdout.write('# of images: %d Current image: %d Elapsed time: %s \t\r' 
                    % (num_images, start_idx, elapsed_time))
            end_idx = min(num_images, start_idx + batch_size)
            images_batch = images[start_idx:end_idx]
            if proc_func:
                images_batch = proc_func(images_batch)
            feed_dict = {self.images: images_batch,
                        self.phase_train: False,
                    self.keep_prob: 1.0}
            mu[start_idx:end_idx], sigma_sq[start_idx:end_idx] = self.sess.run([self.mu, self.sigma_sq], feed_dict=feed_dict)
            # lprint(mu[0, :10])
            # print(sigma_sq[0, :10])
            # exit(0)
        if verbose:
            print('')
        return mu, sigma_sq

    def freeze_model(self, model_dir):
        self.config = imp.load_source('network_config', os.path.join(model_dir, 'config.py'))
        self.initialize(self.config, for_freeze=True)
        with self.sess.graph.as_default():
            var_list = tf.trainable_variables()
            print(len(var_list))
            # model_dir = os.path.expanduser(model_dir)
            # ckpt_file = tf.train.latest_checkpoint(model_dir)
            meta_file, ckpt_file = self.get_model_filenames(model_dir)
            print('Restoring {} variables from {} ...'.format(len(var_list), ckpt_file))
            saver = tf.train.Saver(var_list)
            saver.restore(self.sess, ckpt_file)

            # Retrieve the protobuf graph definition and fix the batch norm nodes
            gd = self.sess.graph.as_graph_def()
            for node in gd.node:
                if node.op == 'RefSwitch':
                    node.op = 'Switch'
                    for index in range(len(node.input)):
                        if 'moving_' in node.input[index]:
                            node.input[index] = node.input[index] + '/read'
                elif node.op == 'AssignSub':
                    node.op = 'Sub'
                    if 'use_locking' in node.attr: del node.attr['use_locking']
                elif node.op == 'AssignAdd':
                    node.op = 'Add'
                    if 'use_locking' in node.attr: del node.attr['use_locking']

            # Get the list of important nodes
            output_node_names = 'mu,sigma_sq'
            whitelist_names = []
            for node in gd.node:
                # if node.name.startswith('InceptionResnetV1') or node.name.startswith('embeddings') or node.name.startswith('phase_train') or node.name.startswith('Bottleneck'):
                print(node.name)
                if not node.name.startswith('Logits'):
                    whitelist_names.append(node.name)

            from tensorflow.python.framework import graph_util
            # Replace all the variables in the graph with constants of the same values
            output_graph_def = graph_util.convert_variables_to_constants(
                self.sess, gd, output_node_names.split(","),
                variable_names_whitelist=whitelist_names)

            # Serialize and dump the output graph to the filesystem
            output_file = os.path.join(model_dir, 'freeze_model.pb')
            with tf.gfile.GFile(output_file, 'wb') as f:
                f.write(output_graph_def.SerializeToString())
            print("%d ops in the final graph." % len(output_graph_def.node))
            print(model_dir)
            print(output_file)


