''' Functions for tensorflow '''
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

import tensorflow as tf
import numpy as np
from tensorflow.python.ops import math_ops, array_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn


def negative_MLS(X, Y, sigma_sq_X, sigma_sq_Y, uncertainty_size):
    print('sigma_sq_X.shape[1] =', sigma_sq_X.shape[1])
    with tf.name_scope('negative_MLS'):
        if uncertainty_size == 1:
            D = X.shape[1].value
            sigma_sq_fuse = sigma_sq_X + tf.transpose(sigma_sq_Y)
            X = tf.stop_gradient(X)
            Y = tf.stop_gradient(Y)
            cos_theta = tf.matmul(X, tf.transpose(Y))
            diffs = 2*(1-cos_theta) / (1e-10 + sigma_sq_fuse) + tf.log(sigma_sq_fuse)
            attention = 2*(1-cos_theta) / (1e-10 + sigma_sq_fuse)
            return diffs, attention
        else:
            D = X.shape[1].value
            X = tf.reshape(X, [-1, 1, D])
            Y = tf.reshape(Y, [1, -1, D])
            sigma_sq_X = tf.reshape(sigma_sq_X, [-1, 1, D])
            sigma_sq_Y = tf.reshape(sigma_sq_Y, [1, -1, D])
            sigma_sq_fuse = sigma_sq_X + sigma_sq_Y
            X_ = tf.stop_gradient(X)
            Y_ = tf.stop_gradient(Y)
            diffs = tf.square(X_-Y_) / (1e-10 + sigma_sq_fuse) + tf.log(sigma_sq_fuse)
            total = tf.reduce_mean(diffs, axis=2)
            attention = tf.reduce_mean(tf.square(X-Y) / (1e-10 + sigma_sq_fuse), axis=2)
            return total, attention


def mutual_likelihood_score_loss(labels, mu, log_sigma_sq, metric_type, uncertainty_size):
    with tf.name_scope('MLS_Loss'):
        batch_size = tf.shape(mu)[0]

        diag_mask = tf.eye(batch_size, dtype=tf.bool)
        non_diag_mask = tf.logical_not(diag_mask)

        sigma_sq = tf.exp(log_sigma_sq)
        print(mu)
        print(sigma_sq)
        loss_mat, attention_mat = negative_MLS(mu, mu, sigma_sq, sigma_sq, uncertainty_size)
        
        label_mat = tf.equal(labels[:,None], labels[None,:])
        label_mask_pos = tf.logical_and(non_diag_mask, label_mat)
        label_mask_neg = tf.logical_and(non_diag_mask, tf.logical_not(label_mat))

        loss_mls = tf.boolean_mask(loss_mat, label_mask_pos)
        attention_pos = tf.boolean_mask(attention_mat, label_mask_pos)
        attention_neg = tf.boolean_mask(attention_mat, label_mask_neg)
        # loss_attention = tf.reduce_mean(attention_pos) - tf.reduce_mean(attention_neg)

        # metric_type = 'contrastive'
        # metric_type = 'triplet_semihard'
        mean_pos = tf.reduce_mean(attention_pos)
        mean_neg = tf.reduce_mean(attention_neg)
        if metric_type == 'contrastive':
            # margin = tf.stop_gradient(mean_pos) + 1  # bad
            # margin = 2.5 #
            margin = 3.0 #
            mean_neg_loss = tf.reduce_mean(tf.maximum(margin - attention_neg, 0))
            # loss_attention = tf.reduce_mean(tf.maximum(mean_pos - attention_neg + 1.0, 0))
            # contrastive_loss
            loss_attention = mean_pos + mean_neg_loss
        elif metric_type == 'triplet_semihard':
            margin = 3.0
            # margin = 1.0 / tf.reduce_mean(sigma_sq)
            loss_attention = triplet_semihard_loss(labels, attention_mat, margin)

        loss_mls = tf.reduce_mean(loss_mls)
        # loss_mls += tf.reduce_mean(loss_mls_neg)
        # loss_mls = tf.reduce_sum(loss_mls) / tf.reduce_sum(tf.cast(loss_mls_neg>0, tf.float32))

        return loss_mls, loss_attention, mean_pos, mean_neg


def center_delta_loss(pred, mu, label, nrof_classes):
    with tf.name_scope("center_delta"):
        D = mu.get_shape()[1]
        centers = tf.get_variable('centers', [nrof_classes, D], dtype=tf.float32,
                                  initializer=tf.constant_initializer(0), trainable=False)
        label = tf.reshape(label, [-1])
        centers_batch = tf.gather(centers, label)
        delta = tf.abs(centers_batch - mu)
        # loss = tf.nn.l2_loss(features - centers_batch)
        loss = tf.reduce_mean(tf.square(pred - delta))
        end_points = {}
        end_points['loss'] = loss
    return loss


def contrastive_loss(labels, embeddings_anchor, embeddings_positive,
                     margin=1.0):
  """Computes the contrastive loss.
  This loss encourages the embedding to be close to each other for
    the samples of the same label and the embedding to be far apart at least
    by the margin constant for the samples of different labels.
  See: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
  Args:
    labels: 1-D tf.int32 `Tensor` with shape [batch_size] of
      binary labels indicating positive vs negative pair.
    embeddings_anchor: 2-D float `Tensor` of embedding vectors for the anchor
      images. Embeddings should be l2 normalized.
    embeddings_positive: 2-D float `Tensor` of embedding vectors for the
      positive images. Embeddings should be l2 normalized.
    margin: margin term in the loss definition.
  Returns:
    contrastive_loss: tf.float32 scalar.
  """
  # Get per pair distances
  distances = math_ops.sqrt(
      math_ops.reduce_sum(
          math_ops.square(embeddings_anchor - embeddings_positive), 1))

  # Add contrastive loss for the siamese network.
  #   label here is {0,1} for neg, pos.
  return math_ops.reduce_mean(
      math_ops.to_float(labels) * math_ops.square(distances) +
      (1. - math_ops.to_float(labels)) *
      math_ops.square(math_ops.maximum(margin - distances, 0.)),
      name='contrastive_loss')


def triplet_semihard_loss(labels, pdist_matrix, margin=1.0):
    """Computes the triplet loss with semi-hard negative mining.
      The loss encourages the positive distances (between a pair of embeddings with
      the same labels) to be smaller than the minimum negative distance among
      which are at least greater than the positive distance plus the margin constant
      (called semi-hard negative) in the mini-batch. If no such negative exists,
      uses the largest negative distance instead.
      See: https://arxiv.org/abs/1503.03832.
      Args:
        labels: 1-D tf.int32 `Tensor` with shape [batch_size] of
          multiclass integer labels.
        embeddings: 2-D float `Tensor` of embedding vectors. Embeddings should
          be l2 normalized.
        margin: Float, margin term in the loss definition.
      Returns:
        triplet_loss: tf.float32 scalar.
    """
    # Reshape [batch_size] label tensor to a [batch_size, 1] label tensor.
    lshape = array_ops.shape(labels)
    assert lshape.shape == 1
    labels = array_ops.reshape(labels, [lshape[0], 1])

    # Build pairwise squared distance matrix.
    # pdist_matrix = pairwise_distance(embeddings, squared=True)
    # Build pairwise binary adjacency matrix.
    adjacency = math_ops.equal(labels, array_ops.transpose(labels))
    # Invert so we can select negatives only.
    adjacency_not = math_ops.logical_not(adjacency)

    batch_size = array_ops.size(labels)

    # Compute the mask.
    pdist_matrix_tile = array_ops.tile(pdist_matrix, [batch_size, 1])
    mask = math_ops.logical_and(
        array_ops.tile(adjacency_not, [batch_size, 1]),
        math_ops.greater(
            pdist_matrix_tile, array_ops.reshape(
                array_ops.transpose(pdist_matrix), [-1, 1])))
    mask_final = array_ops.reshape(
        math_ops.greater(
            math_ops.reduce_sum(
                math_ops.cast(
                    mask, dtype=dtypes.float32), 1, keep_dims=True),
            0.0), [batch_size, batch_size])
    mask_final = array_ops.transpose(mask_final)

    adjacency_not = math_ops.cast(adjacency_not, dtype=dtypes.float32)
    mask = math_ops.cast(mask, dtype=dtypes.float32)

    # negatives_outside: smallest D_an where D_an > D_ap.
    negatives_outside = array_ops.reshape(
        masked_minimum(pdist_matrix_tile, mask), [batch_size, batch_size])
    negatives_outside = array_ops.transpose(negatives_outside)

    # negatives_inside: largest D_an.
    negatives_inside = array_ops.tile(
        masked_maximum(pdist_matrix, adjacency_not), [1, batch_size])
    semi_hard_negatives = array_ops.where(
        mask_final, negatives_outside, negatives_inside)

    loss_mat = math_ops.add(margin, pdist_matrix - semi_hard_negatives)

    mask_positives = math_ops.cast(
        adjacency, dtype=dtypes.float32) - array_ops.diag(
        array_ops.ones([batch_size]))

    # In lifted-struct, the authors multiply 0.5 for upper triangular
    #   in semihard, they take all positive pairs except the diagonal.
    num_positives = math_ops.reduce_sum(mask_positives)

    _triplet_loss = math_ops.truediv(
        math_ops.reduce_sum(
            math_ops.maximum(
                math_ops.multiply(loss_mat, mask_positives), 0.0)),
        num_positives,
        name='triplet_semihard_loss')
    return _triplet_loss


def pairwise_distance(feature, squared=False):
    """Computes the pairwise distance matrix with numerical stability.
    output[i, j] = || feature[i, :] - feature[j, :] ||_2
    Args:
        feature: 2-D Tensor of size [number of data, feature dimension].
        squared: Boolean, whether or not to square the pairwise distances.
    Returns:
        pairwise_distances: 2-D Tensor of size [number of data, number of data].
    """
    pairwise_distances_squared = math_ops.add(
        math_ops.reduce_sum(
            math_ops.square(feature),
            axis=[1],
            keep_dims=True),
        math_ops.reduce_sum(
            math_ops.square(
                array_ops.transpose(feature)),
            axis=[0],
            keep_dims=True)) - 2.0 * math_ops.matmul(
        feature, array_ops.transpose(feature))

    # Deal with numerical inaccuracies. Set small negatives to zero.
    pairwise_distances_squared = math_ops.maximum(pairwise_distances_squared, 0.0)
    # Get the mask where the zero distances are at.
    error_mask = math_ops.less_equal(pairwise_distances_squared, 0.0)

    # Optionally take the sqrt.
    if squared:
        pairwise_distances = pairwise_distances_squared
    else:
        pairwise_distances = math_ops.sqrt(
            pairwise_distances_squared + math_ops.to_float(error_mask) * 1e-16)

    # Undo conditionally adding 1e-16.
    pairwise_distances = math_ops.multiply(
        pairwise_distances, math_ops.to_float(math_ops.logical_not(error_mask)))

    num_data = array_ops.shape(feature)[0]
    # Explicitly set diagonals to zero.
    mask_offdiagonals = array_ops.ones_like(pairwise_distances) - array_ops.diag(
        array_ops.ones([num_data]))
    pairwise_distances = math_ops.multiply(pairwise_distances, mask_offdiagonals)
    return pairwise_distances


def masked_minimum(data, mask, dim=1):
    """Computes the axis wise minimum over chosen elements.
  Args:
    data: 2-D float `Tensor` of size [n, m].
    mask: 2-D Boolean `Tensor` of size [n, m].
    dim: The dimension over which to compute the minimum.
  Returns:
    masked_minimums: N-D `Tensor`.
      The minimized dimension is of size 1 after the operation.
  """
    axis_maximums = math_ops.reduce_max(data, dim, keep_dims=True)
    masked_minimums = math_ops.reduce_min(
        math_ops.multiply(
            data - axis_maximums, mask), dim, keep_dims=True) + axis_maximums
    return masked_minimums

def masked_maximum(data, mask, dim=1):
    """Computes the axis wise maximum over chosen elements.
      Args:
        data: 2-D float `Tensor` of size [n, m].
        mask: 2-D Boolean `Tensor` of size [n, m].
        dim: The dimension over which to compute the maximum.
      Returns:
        masked_maximums: N-D `Tensor`.
          The maximized dimension is of size 1 after the operation.
    """
    axis_minimums = math_ops.reduce_min(data, dim, keep_dims=True)
    masked_maximums = math_ops.reduce_max(
        math_ops.multiply(
            data - axis_minimums, mask), dim, keep_dims=True) + axis_minimums
    return masked_maximums


