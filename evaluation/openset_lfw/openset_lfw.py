import numpy as np
import os
import scipy.io as sio
from numpy.linalg import norm
import time

from evaluation.openset_lfw.openset_roc import OpenSetROC
from utils.utils import pair_cosin_score, nvm_MLS_score, nvm_MLS_score_attention

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
# root_dir = r'G:\chenkai\ProbFace\proto/'
# root_dir = os.path.join(CUR_DIR, '..', '..', 'proto')
project_root = os.path.dirname(os.path.dirname(CUR_DIR))
proto_root_dir = project_root + '/proto/'
# configFile = root_dir + 'data/blufr/blufr_lfw_config.mat'  # configuration file for this evaluation
# data = sio.loadmat(configFile)

veriFarPoints = np.array([0] + list(np.kron([10 ** l for l in range(-8, 0)], range(1, 10))) + [
    1])  # FAR points for face verification ROC plot
osiFarPoints = np.array([0] + list(np.kron([10 ** l for l in range(-4, 0)], range(1, 10))) + [
    1])  # FAR points for open-set face identification ROC plot
rankPoints = np.array(
    list(range(10)) + list(range(20, 101, 10)))  # rank points for open-set face identification CMC plot
reportVeriFar = 0.001  # the FAR point for verification performance reporting
reportOsiFar = 0.01  # the FAR point for open-set identification performance reporting
reportRank = 1  # the rank point for open-set identification performance reporting


def openset_lfw(feats, compare_func, numTrials=None):
    start_time = time.time()

    data = sio.loadmat(proto_root_dir + 'blufr/id_lfw.mat')
    id_lfw = data['id_lfw']

    data = sio.loadmat(proto_root_dir + 'blufr/lfw_gallery_index.mat')
    gallery_index = np.array([
        data['gallery_index_1'].ravel(),
        data['gallery_index_2'].ravel(),
        data['gallery_index_3'].ravel(),
        data['gallery_index_4'].ravel(),
        data['gallery_index_5'].ravel(),
        data['gallery_index_6'].ravel(),
        data['gallery_index_7'].ravel(),
        data['gallery_index_8'].ravel(),
        data['gallery_index_9'].ravel(),
        data['gallery_index_10'].ravel(),
        ])
    gallery_index -= 1

    data = sio.loadmat(proto_root_dir + 'blufr/lfw_genuine_probe_index.mat')
    genuine_probe_index = np.array([
        data['genuine_probe_index_1'].ravel(),
        data['genuine_probe_index_2'].ravel(),
        data['genuine_probe_index_3'].ravel(),
        data['genuine_probe_index_4'].ravel(),
        data['genuine_probe_index_5'].ravel(),
        data['genuine_probe_index_6'].ravel(),
        data['genuine_probe_index_7'].ravel(),
        data['genuine_probe_index_8'].ravel(),
        data['genuine_probe_index_9'].ravel(),
        data['genuine_probe_index_10'].ravel(),
        ])
    genuine_probe_index -= 1

    data = sio.loadmat(proto_root_dir + 'blufr/lfw_impostor_probe_index.mat')
    impostor_probe_index = np.array([
        data['impostor_probe_index_1'].ravel(),
        data['impostor_probe_index_2'].ravel(),
        data['impostor_probe_index_3'].ravel(),
        data['impostor_probe_index_4'].ravel(),
        data['impostor_probe_index_5'].ravel(),
        data['impostor_probe_index_6'].ravel(),
        data['impostor_probe_index_7'].ravel(),
        data['impostor_probe_index_8'].ravel(),
        data['impostor_probe_index_9'].ravel(),
        data['impostor_probe_index_10'].ravel(),
        ])
    impostor_probe_index -= 1

    # You may apply the sqrt transform if the feature is histogram based.
    # Descriptors = sqrt(double(Descriptors))
    Descriptors = feats

    if numTrials is None:
        numTrials = gallery_index.shape[0]

    numVeriFarPoints = len(veriFarPoints)
    VR = np.zeros((numTrials, numVeriFarPoints)) # verification rates of the 10 trials
    veriFAR = np.zeros((numTrials, numVeriFarPoints)) # verification false accept rates of the 10 trials

    numOsiFarPoints = len(osiFarPoints)
    numRanks = len(rankPoints)
    DIR = np.zeros((numRanks, numOsiFarPoints, numTrials)) # detection and identification rates of the 10 trials
    osiFAR = np.zeros((numTrials, numOsiFarPoints)) # open-set identification false accept rates of the 10 trials
    thred = np.zeros((numTrials, numOsiFarPoints)) # open-set identification false accept rates of the 10 trials

    ## Get the FAR or rank index where we report performance.
    # [~, veriFarIndex] = ismember(reportVeriFar, veriFarPoints)
    veriFarIndex = np.argmax(reportVeriFar == veriFarPoints)
    # [~, rankIndex] = ismember(reportRank, rankPoints)
    rankIndex = np.argmax(reportRank == rankPoints)

    # fprintf('Evaluation with 10 trials.\n\n')

    ## Evaluate with 10 trials.
    for t in range(numTrials):
        # fprintf('Process the #dth trial...\n\n', t)

        # Get the training data of the t'th trial.
        # X = Descriptors(trainIndex{t}, :)

        feats_g = feats[gallery_index[t,:],:]
        labels_g = id_lfw[gallery_index[t,:]]

        feats_pg = feats[genuine_probe_index[t,:],:]
        labels_pg = id_lfw[genuine_probe_index[t,:]]

        feats_pn = feats[impostor_probe_index[t,:],:]
        labels_pn = id_lfw[impostor_probe_index[t,:]]

        X1 = feats_g
        X2 = np.concatenate((feats_pg, feats_pn), axis=0)

        # Transform the test data into the learned PCA subspace of pcaDims dimensions.
        # X = X * W(:, 1 : pcaDims)

        # Normlize each row to unit length. If you do not have this function,
        # do it manually.
        X1 = np.array([l/norm(l) for l in X1])
        X2 = np.array([l/norm(l) for l in X2])

        # print(t)
        # Compute the cosine similarity score between the test samples.
        t1 = time.time()
        if compare_func == pair_cosin_score:
            score_array = np.dot(X1, X2.T)
        elif compare_func == nvm_MLS_score or compare_func == nvm_MLS_score_attention or True:
            # score_array = nvm_MLS_score(X1, X2)
            score_array = []
            batch_size = X1.shape[0]
            for start_idx in range(0, X1.shape[0], batch_size):
                # print(start_idx)
                end_idx = min(X1.shape[0], start_idx + batch_size)
                score_array += list(compare_func(X1[start_idx:end_idx], X2))
        else:
            score_array = []
            for l1 in X1:
                L1 = np.repeat([l1], X2.shape[0], axis=0)
                score_array += [compare_func(L1, X2)]
                # print(t, np.array(score_array).shape, X1.shape, X2.shape)
                # import sys
                # sys.stdout.write('{} {} {} {} \t\r'.format(t, np.array(score_array).shape, X1.shape, X2.shape))
            # print('')
        # print('--> ', compare_func, 'n', len(X1)*len(X2), 'time', time.time()-t1)

        score = np.array(score_array)
        # print(score.shape)

        # negative L2 distance score
        # normX1 = sum(X1.^2,2)
        # normX2 = sum(X2.^2,2)
        # score = repmat(normX1 ,1,size(X2,1)) + repmat(normX2',size(X1,1),1) - 2*score_cos
        # score = -score

        # Get the class labels for the test data of the development set.
        labels_p = np.concatenate((labels_pg, labels_pn), axis=0)

        # Evaluate the open-set identification performance.
        DIR_, osiFAR_, thred_ = OpenSetROC(score, labels_g, labels_p, osiFarPoints, rankPoints)
        DIR[:,:,t] = DIR_
        osiFAR[t,:] = osiFAR_
        thred[t,:] = thred_

    ## Average over the 10 trials, and compute the standard deviation.

    # testTime = toc
    # fprintf('\nEvaluation time: #.0f seconds.\n', testTime)

    # print('gallery        #person %6d  #images %6d' % (len(np.unique(labels_g)), len(labels_g)))
    # print('genuine probe  #person %6d  #images %6d' % (len(np.unique(labels_pg)), len(labels_pg)))
    # print('impostor probe #person %6d  #images %6d' % (len(np.unique(labels_pn)), len(labels_pn)))

    s = 'Open-set Identification:\n'
    # for reportOsiFar = [1.0 0.9 0.8 0.4 0.2 0.1 0.01 0.001 0.0005 0.0001]
    for reportOsiFar in [0.01, 0.001, 0.0005, 0.0004, 0.0003, 0.0002, 0.0001]:
    # for reportOsiFar in [0.01, 0.001, 0.0005, 0.0001]:
        # [~, osiFarIndex] = ismember(reportOsiFar, osiFarPoints)
        # osiFarIndex = np.argmax(reportOsiFar == osiFarPoints)
        osiFarIndex = np.sum(reportOsiFar > osiFarPoints)
        meanThred = np.mean(thred, axis=0)
        meanOsiFAR = np.mean(osiFAR, axis=0)
        meanDIR = np.mean(DIR, axis=2)
        reportMeanDIR = meanDIR[rankIndex, osiFarIndex]
        reportThred = meanThred[osiFarIndex]

        ## Display the benchmark performance and output to the log file.
        s_far = '%g%%' % (reportOsiFar*100)
        s += '\t@ Rank = %d, FAR = %6s, Thred=%.2f, DIR = %.2f%%.\n' % (reportRank, s_far, reportThred, reportMeanDIR*100)
    # print(s)
    elapsed_time = time.strftime('%H:%M:%S', time.gmtime(time.time()-start_time))
    print('openset_lfw elapsed_time', elapsed_time)
    return s


def get_paths(lfw_dir, lfw_pairs_file, file_ext):
    pairs = []
    with open(lfw_pairs_file, 'r') as f:
        for line in f.readlines()[1:]:
            pair = line.strip().split()
            pairs.append(pair)
    nrof_skipped_pairs = 0
    path_list = []
    issame_list = []
    for pair in pairs:
        if len(pair) == 3:
            path0 = os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])+'.'+file_ext)
            path1 = os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[2])+'.'+file_ext)
            issame = True
        elif len(pair) == 4:
            path0 = os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])+'.'+file_ext)
            path1 = os.path.join(lfw_dir, pair[2], pair[2] + '_' + '%04d' % int(pair[3])+'.'+file_ext)
            issame = False
        if os.path.exists(path0) and os.path.exists(path1):    # Only add the pair if both paths exist
            path_list += (path0, path1)
            issame_list.append(issame)
        else:
            nrof_skipped_pairs += 1
    if nrof_skipped_pairs > 0:
        print('Skipped %d image pairs' % nrof_skipped_pairs)

    return path_list, issame_list

def get_paths_all(lfw_dir):
    pairs = os.path.join(proto_root_dir, 'lfw_pairs.txt')
    file_ext = 'jpg'
    path_WDRef = os.path.join(proto_root_dir, 'WDRef')
    data = sio.loadmat(path_WDRef + '/imagelist_lfw')
    imagelist_lfw = data['imagelist_lfw'].ravel()
    path_list = []
    for i, filename in enumerate(imagelist_lfw):
        filename = filename[0]
        filename = filename.replace('\\','/')
        full_path = os.path.join(lfw_dir, filename)
        path_list += [full_path]
    pair_path_list, pair_issame_list = get_paths(lfw_dir, pairs, file_ext)
    return path_list



if __name__ == '__main__':
    # feaFile = 'data/LightenedCNN_B_lfw.mat'
    # load(feaFile)
    # feats = features
    # load('data/index_lightcnn_2_wdref.mat')
    # feats = feats(index_lightcnn_2_wdref,:)
    feaFile = r'F:\docker-images\faceid_deploy\evaluate\evaluate_lfw\feats_lfw.npy'
    if feaFile.endswith('.mat'):
        data = sio.loadmat(feaFile)
        feats = data['feats']
    else:
        feats = np.load(feaFile)
    print(feaFile)
    print(openset_lfw(feats, nvm_MLS_score))
