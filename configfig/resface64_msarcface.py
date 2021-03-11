''' Config Proto '''

import sys
import os


####### INPUT OUTPUT #######

# The name of the current model for output
name = 'resface64'

# The folder to save log and model
log_base_dir = './log/'

# The interval between writing summary
summary_interval = 100

# Training dataset path
train_dataset_path = r"F:\data\face-recognition\trillion-pairs\challenge\ms1m-retinaface-t1-img"

test_data_dir_mx = r'F:\data\face-recognition\trillion-pairs\challenge\ms1m-retinaface-t1'

# Target image size for the input of network
image_size = [96, 96]

# 3 channels means RGB, 1 channel for grayscale
channels = 3

# Preprocess for training
preprocess_train = [
    # ['center_crop', (image_size[0], image_size[1])],
    ['resize', (image_size[0], image_size[1])],
    ['random_flip'],
    ['standardize', 'mean_scale'],
]

# Preprocess for testing
preprocess_test = [
    ['resize', (image_size[0], image_size[1])],
    ['standardize', 'mean_scale'],
]

# Number of GPUs
num_gpus = 1

####### NETWORK #######

# The network architecture
embedding_network = "models/resface64_relu.py"

# The network architecture
uncertainty_module = "models/uncertainty_module.py"
uncertainty_module_input = "conv_final"

# Number of dimensions in the embedding space
embedding_size = 256

# uncertainty_module_output_size = embedding_size
uncertainty_module_output_size = 1


####### TRAINING STRATEGY #######

# Base Random Seed
base_random_seed = 0

# Number of samples per batch
batch_size = 128
batch_format = {
    'size': batch_size,
    'num_classes': batch_size // 4,
}

# Number of batches per epoch
epoch_size = 1000

# Number of epochs
# num_epochs = 32
num_epochs = 12

# learning rate strategy
learning_rate_strategy = 'step'

# learning rate schedule
# lr = 3e-3
lr = 1e-2
learning_rate_schedule = {
    0:      1 * lr,
    num_epochs/4*2000:   0.1 * lr,
    num_epochs/4*3000:   0.1 * lr,
}

# Restore model
restore_model = r'log/resface64/20200122-214343-arc'

# Keywords to filter restore variables, set None for all
# restore_scopes = ['Resface']
# restore_scopes = ['StudentUncertaintyModule']
# restore_scopes = ['Resface', 'UncertaintyModule']
restore_scopes = None
# exclude_restore_scopes = ['center_loss', 'centers']
# exclude_restore_scopes = ['center_loss', 'centers', 'fc_log_sigma_sq']
exclude_restore_scopes = ['center_loss', 'centers', 'UncertaintyModule']

# Weight decay for model variables
weight_decay = 5e-4

# Keep probability for dropouts
keep_prob = 1.0

# discriminat_metric_type = 'contrastive'
discriminat_metric_type = 'triplet_semihard'

loss_weights = {
    'mls_loss': 1.0,
    'output_constraint_loss': 0.001,
    'discriminate_loss': 0.001,
}

output_constraint_type = 'L2'
# output_constraint_type = 'L1'
# s_wd_type = ''
# s_wd_alpha = 0.001

