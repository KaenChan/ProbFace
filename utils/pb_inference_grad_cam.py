# -*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from tensorflow.python.platform import gfile
import os
import sys
import shutil
import copy


class pbInference(object):
    def __init__(self, PATH_TO_PB, WIDTH_HEIGH, gpu_num='-1', input_name='input:0', output_name='output:0'):
        pid = os.getpid()
        print('------init pid-----', pid)
        self.WIDTH_HEIGH = WIDTH_HEIGH
        self.input_name = input_name
        self.output_name = output_name
        self.sess = self.config(gpu_num=gpu_num)
        # self.sess = tf.Session(graph=g)
        with gfile.FastGFile(PATH_TO_PB, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            self.sess.graph.as_default()
            tf.import_graph_def(graph_def, name='')  # 导入计算图
        self.sess.run(tf.global_variables_initializer())

    def config(self, gpu_num='-1'):
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_num
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        session = tf.Session(config=config)
        # KTF.set_session(session)
        return session

    def load_image_into_numpy_array(self, image):
        (im_width, im_height) = image.size
        return np.array(image.getdata()).reshape(
            (im_height, im_width, 3)).astype(np.float32)
        # return np.array(image.getdata()).reshape(
        #     (im_height, im_width, 3)).astype(np.uint8)

    def returnCAM(self, img, logits_layer, conv_layer, pred_class, num_classes=2):
        if num_classes != 1:
            one_hot = tf.sparse_to_dense(pred_class, [num_classes], 1.0)
            signal = tf.multiply(logits_layer, one_hot)
        else:
            signal = -logits_layer
        loss = tf.reduce_mean(signal)

        grads = tf.gradients(loss, conv_layer)[0]
        # Normalizing the gradients
        norm_grads = tf.divide(grads, tf.sqrt(tf.reduce_mean(tf.square(grads))) + tf.constant(1e-5))

        input = self.sess.graph.get_tensor_by_name(self.input_name)
        output, grads_val = self.sess.run([conv_layer, norm_grads], feed_dict={input: img})
        output = output[0]  # [10,10,2048]
        grads_val = grads_val[0]  # [10,10,2048]

        weights = np.mean(grads_val, axis=(0, 1))  # [2048]
        cam = np.ones(output.shape[0: 2], dtype=np.float32)  # [10,10]

        # Taking a weighted average
        for i, w in enumerate(weights):
            cam += w * output[:, :, i]

        # Passing through ReLU
        cam = np.maximum(cam, 0)
        cam = cam / np.max(cam)
        # cam = cv2.resize(cam, (img.shape[1], img.shape[2]))
        return cam

    def service(self, image_path, feature_name=None):
        image = Image.open(image_path)
        # image = image.convert('L').convert('RGB')  # 转3通道灰度图
        image_np = self.load_image_into_numpy_array(image)
        image_np_orig = copy.copy(image_np)
        image_np_orig = np.uint8(image_np_orig)
        # image_np /= 255.
        image_np = (image_np - 128.) / 128.
        image_np = cv2.resize(image_np, (self.WIDTH_HEIGH, self.WIDTH_HEIGH))
        # cv2.imshow('image_np', image_np)
        # cv2.waitKey(0)
        image_np_expanded = np.expand_dims(image_np, axis=0)

        input = self.sess.graph.get_tensor_by_name(self.input_name)
        label = self.sess.graph.get_tensor_by_name(self.output_name)

        if feature_name is not None and len(feature_name):
            # feature_name = 'net/Resface/Conv4/Conv4/Conv4_1/add'
            feature = self.sess.graph.get_operation_by_name(feature_name).outputs[0]
            output_dict, feature_map = self.sess.run([label, feature], feed_dict={input: image_np_expanded})
            print('output_dict:', output_dict)

            label = np.array(output_dict[0]).argmax()
            score = output_dict[0][label]

            # generate class activation mapping for the top1 prediction
            # logits_name = 'net/live_clf/fully_connected/BiasAdd'
            # logits = self.sess.graph.get_operation_by_name(logits_name).outputs[0]
            logits = self.sess.graph.get_tensor_by_name(self.output_name)
            print('returnCAM..')
            cam_img = self.returnCAM(image_np_expanded, logits, feature, label, num_classes=1)
            print('returnCAM..end')
            cam_img = cv2.resize(cam_img, (image_np_orig.shape[1], image_np_orig.shape[0]))
            cam_img = cam_img.astype(float)
            cam_img /= cam_img.max()
            cam_img = np.uint8(255 * cam_img)
            cam3 = cv2.applyColorMap(cam_img, cv2.COLORMAP_JET)
            # cam3 = cv2.cvtColor(cam3, cv2.COLOR_BGR2RGB)
            frame_out = cv2.addWeighted(image_np_orig[...,::-1], 0.7, cam3, 0.3, 0)
            frame_out = cv2.resize(frame_out, (400, 400))
            cv2.imshow('frame_out', frame_out)
            cv2.waitKey(1)

            # feat_img = feature_map[0, :, :, 1]
            # feat_img = cv2.resize(feat_img, (image_np_orig.shape[1], image_np_orig.shape[0]), interpolation=cv2.INTER_CUBIC)
            # feat_img = np.uint8(255 * feat_img)
            # feat_img = cv2.cvtColor(feat_img, cv2.COLOR_GRAY2BGR)
            # heatmap1 = cv2.applyColorMap(feat_img, cv2.COLORMAP_JET)
            # image_np_orig = np.uint8(image_np_orig)
            # frame_out = cv2.addWeighted(image_np_orig, 0.7, heatmap1, 0.3, 0)
            # cv2.imshow('feat_img', feat_img)
            # cv2.imshow('frame_out', frame_out)
            # cv2.waitKey(0)

            # with tf.GradientTape() as gtape:
            #     prob = label[:, np.argmax(label[0])]  # 最大可能性类别的预测概率
            #     grads = gtape.gradient(prob, feature)  # 类别与卷积层的梯度 (1,14,14,512)
            #     print('grads', grads)
            #     pooled_grads = np.mean(grads, axis=(0, 1, 2))  # 特征层梯度的全局平均代表每个特征层权重
            # heatmap = tf.reduce_mean(tf.multiply(pooled_grads, feature), axis=-1)  # 权重与特征层相乘，512层求和平均
            # # print('heatmap',heatmap)
            # print(heatmap.shape)
            # heatmap = np.maximum(heatmap, 0)
            # heatmap1 = cv2.resize(heatmap[0], (image_np_orig.shape[1], image_np_orig.shape[0]),
            #                       interpolation=cv2.INTER_CUBIC)
            # heatmap1 = np.uint8(255 * heatmap1)
            # heatmap1 = cv2.applyColorMap(heatmap1, cv2.COLORMAP_JET)
            # frame_out = cv2.addWeighted(image_np_orig, 0.5, heatmap1, 0.5, 0)
            # cv2.imshow('frame_out', frame_out)

            # label = np.array(output_dict[0]).argmax()
            # score = output_dict[0][label]

            return label, score, frame_out
        else:
            output_dict = self.sess.run(label, feed_dict={input: image_np_expanded})
            print('output_dict:', output_dict)
            label = np.array(output_dict[0]).argmax()
            score = output_dict[0][label]

        return label, score


def infer_dirs(g):
    # root_dirs = ['/media/zachary/diske/data/faces/xiaogu-videos-fake/face_112x112/sn_dual_nir/jiguang/7',
    #              '/media/zachary/diske/data/faces/xiaogu-videos-fake/face_112x112/sn_dual_nir/jiguang-w/7',
    #              '/media/zachary/diske/data/faces/xiaogu-videos-live/face_112x112/sn_dualIR/nir/cam0']

    # root_dirs = ['/media/zachary/diske/data/faces/xiaogu-videos-fake/face_112x112/sn_dual_nir/jiguang/7',
    #              '/media/zachary/diske/data/faces/xiaogu-videos-fake/face_112x112/sn_dual_nir/jiguang-w/7']

    # root_dirs = ['/media/zachary/diske/data/faces/xiaogu-videos-fake/face_112x112/sn_dual_nir']
    # root_dirs = ['/media/zachary/diske/data/faces/xiaogu-videos-live-persons-quality/shoulder_head_400x400']
    # root_dirs = ['/media/zachary/diske/data/faces/xiaogu-videos-fake-test/dataset_0813to14/shoulder_head_400x400/sn_dual_nir_0813']
    # root_dirs = ['/media/zachary/diske/data/faces/xiaogu-videos-fake-persons/shoulder_head_400x400']
    # root_dirs = ['/media/zachary/diske/ubuntu-workspace/Face_Detection_Alignment-djk/test_mini3/image_20200905_test_camera_shead']
    # root_dirs = [r'F:\data\face-recognition\lfw\lfw-112-mxnet']
    root_dirs = [r'F:\data\face-recognition\lfw\lfw-112-mxnet-distorted\gaussian_blur']
    root_dirs = [r'F:\data\face-recognition\test\IJB-A\CleanData_mxnet_96_skimage_sigma0.008\img']

    badcase_dir = './badcase/'

    total = 0
    correct = 0
    with open('./result_fake_all.txt', 'w') as f:
        with open('./result_fake_all_badcase.txt', 'w') as fb:
            for i, root_dir in enumerate(root_dirs):
                for root, dirs, files in os.walk(root_dir, followlinks=True):
                    for img_file in files[:70]:
                        if not img_file.endswith('jpg'):
                            continue
                        # if '/7' in root:
                        #     continue
                        # if 'test_mini3' in root and '400_400' not in img_file:
                        #     continue
                        img_path_file = os.path.join(root, img_file)
                        print(img_path_file)
                        # label, score = g.service(img_path_file)
                        label, score, feature_map = g.service(img_path_file, feature_name='Resface/Conv4/Conv_4/Conv_4_3/add')
                        print('label:', label)
                        print('score:', score)
                        if 'live' in root:
                            gt_label = 0
                        else:
                            gt_label = 1

                        out_dir = r'G:\chenkai\Probabilistic-Face-Embeddings-master\utils\cam_result\probface_neg_sigma_70'
                        if not os.path.exists(out_dir):
                            os.makedirs(out_dir)
                        dst_img_path = os.path.join(out_dir, '%.2f_' % score + img_file)
                        print(dst_img_path)
                        # shutil.copy(img_path_file, dst_img_path)
                        cv2.imwrite(dst_img_path, feature_map)

                        total += 1
                        if int(label) == gt_label:
                            correct += 1
                        # else:
                            # sub_dir = root.split('/media/zachary/diske/data/faces/')[1]
                            # out_dir = os.path.join(badcase_dir, sub_dir)
                            # if not os.path.exists(out_dir):
                            #     os.makedirs(out_dir)
                            # dst_img_path = os.path.join(out_dir, img_file)
                            # print(dst_img_path)
                            # shutil.copy(img_path_file, dst_img_path)
                            # fb.write('\n{},{},{},{}'.format(dst_img_path, str(gt_label), str(label), str(score)))

                        # f.write('\n{},{},{},{}'.format(img_path_file, str(gt_label), str(label), str(score)))

                        # if total > 10:
                        #     sys.exit(0)

        print('accuracy={}, correct={}, total={}'.format(correct / total, correct, total))
        """
        fake: accuracy=0.999228279277155, correct=46613, total=46649
        fake-all: accuracy=0.9996306945580206, correct=303160, total=303272
        """


if __name__ == '__main__':
    # t1 = time.time()
    PATH_TO_PB = r'G:\chenkai\Probabilistic-Face-Embeddings-master\log\resface64_relu_msarcface_am_PFE\20201020-180059-mls-only\freeze_model.pb'
    PATH_TO_PB = r'G:\chenkai\Probabilistic-Face-Embeddings-master\log\resface64m_relu_msarcface_am_PFE\20201028-193142-mls1.0-abs0.001-triplet0.0001\freeze_model.pb'
    g = pbInference(PATH_TO_PB=PATH_TO_PB, WIDTH_HEIGH=96, gpu_num='-1', input_name='input:0',
                    output_name='sigma_sq:0')
    # label, score = g.service(file)
    # print('label:', label)
    # print('score:', score)

    infer_dirs(g)







