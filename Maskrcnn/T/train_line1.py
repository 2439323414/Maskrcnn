# -*- coding: utf-8 -*-
# 用于训练权重
"""
2020.09.25：alian
标志标线检测模型训练，类别：边线、路面标志
"""
import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
from mrcnn.config import Config
# import utils
# from mrcnn import utils
from mrcnn import model as modellib, utils  # 等价于from mrcnn import utils
from mrcnn import visualize
import yaml
import cv2
from mrcnn.model import log
from PIL import Image

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'   #指定第一块GPU可用
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.5  # 程序最多只能占用指定gpu50%的显存
# config.gpu_options.allow_growth = True      #程序按需申请内存
# sess = tf.Session(config = config)

# 获得当前工作目录
ROOT_DIR = os.getcwd()

# ROOT_DIR = os.path.abspath("../") 获得绝对路径
# 保存logs和训练模型的文件
MODEL_DIR = os.path.join(ROOT_DIR, "logs")  #新建logs文件夹

iter_num = 0

# 权重文件路径
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_balloon.h5")  # 预训练权重文件
# 下载COCO预训练模型
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)


class ShapesConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # 为配置制定一个可识别的名字
    NAME = "line"

    # 每个GPU训练的图片数，TBatch size=GPUs * images/GPU)如果gpu允许的话可以设置大些
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # 类别数量包括背景
    NUM_CLASSES = 1 + 1

    # 定义图像大小
    # 至少被2的6次方整除512,832
    IMAGE_MIN_DIM = 800
    IMAGE_MAX_DIM = 1024

    # 锚框
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)  # anchor side in pixels
    RPN_ANCHOR_RATIOS = [0.5, 1, 2]
    # 每个图像的ROI数量
    TRAIN_ROIS_PER_IMAGE = 200

    # epoch：每个时期的训练步数，不需要与训练集一致，每个人时期（epoch）未保存的Tensorboard以及计算验证统计信息
    STEPS_PER_EPOCH = 24

    # 每个训练时期结束时运行的验证数，较大的数字可以提高验证统计数据的准确性，但会降低训练速度
    VALIDATION_STEPS = 16


config = ShapesConfig()
config.display()

# 训练集
class DrugDataset(utils.Dataset):
    # 得到该图中有多少个实例（物体）
    def get_obj_index(self, image):
        n = np.max(image)
        return n

    # 解析labelme中得到的yaml文件，从而得到mask每一层对应的实例标签
    def from_yaml_get_class(self, image_id):
        info = self.image_info[image_id]
        with open(info['yaml_path']) as f:
            temp = yaml.load(f.read(), Loader=yaml.FullLoader)
            labels = temp['label_names']
            del labels[0]
        return labels

    # 重新写draw_mask
    def draw_mask(self, num_obj, mask, image, image_id):
        info = self.image_info[image_id]
        for index in range(num_obj):
            for i in range(info['width']):
                for j in range(info['height']):
                    # print("image_id-->",image_id,"-i--->",i,"-j--->",j)
                    # print("info[width]----->",info['width'],"-info[height]--->",info['height'])
                    at_pixel = image.getpixel((i, j))
                    if at_pixel == index + 1:
                        mask[j, i, index] = 1
        return mask

    # 重新写load_shapes，里面包含自己的类别,可以任意添加
    # 并在self.image_info信息中添加了path、mask_path 、yaml_path
    dataset_root_path = r"E:\OCR_picture"
   # yaml_pathdataset_root_path = r"E:\OCR_picture"
    img_floder = dataset_root_path + "rgb"
    mask_floder = dataset_root_path + "mask"

    def load_shapes(self, count, img_floder, mask_floder, imglist, dataset_root_path):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        # Add classes,可通过这种方式扩展多个物体
        self.add_class("shapes", 1, "word")  # 标线
        self.add_class("shapes", 2, "picture") # 路面标志
        self.add_class("shapes", 3, "table")
        self.add_class("shapes", 4, "title")
        for i in range(train):
            # 获取图片宽和高

            filestr = imglist[i].split(".")[0]
            # filestr = filestr.split("_")[1]
            mask_path = mask_floder + filestr + "_json/" + "label.png"
            #yaml_path = mask_floder + filestr + "_json/info.yaml"
            cv_img = cv2.imread(mask_floder + filestr + "_json/" + "img.png")
            print(img_floder + "/" + imglist[i])

            self.add_image("shapes", image_id=i, path=img_floder + "/" + imglist[i],
                           width=cv_img.shape[1], height=cv_img.shape[0], mask_path=mask_path, yaml_path=yaml_path)

    # 重写load_mask
    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        global iter_num
        print("image_id", image_id)
        info = self.image_info[image_id]
        count = 1  # number of object
        img = Image.open(info['mask_path'])
        num_obj = self.get_obj_index(img)
        mask = np.zeros([info['height'], info['width'], num_obj], dtype=np.uint8)
        mask = self.draw_mask(num_obj, mask, img, image_id)
        occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
        for i in range(count - 2, -1, -1):
            mask[:, :, i] = mask[:, :, i] * occlusion

            occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))

        labels = self.from_yaml_get_class(image_id)
        labels_form = []
        for i in range(len(labels)):
            if labels[i].find("word") != -1:
                labels_form.append("word")
            elif labels[i].find("picture") != -1:
                labels_form.append("picture")
            elif labels[i].find("table") != -1:
                labels_form.append("table")
            elif labels[i].find("title") != -1:
                labels_form.append("title")
        class_ids = np.array([self.class_names.index(s) for s in labels_form])
        return mask, class_ids.astype(np.int32)

# 验证集
class DrugDataset_val(utils.Dataset):
    # 得到该图中有多少个实例（物体）
    def get_obj_index(self, image):
        n = np.max(image)
        return n

    # 解析labelme中得到的yaml文件，从而得到mask每一层对应的实例标签
    def from_yaml_get_class(self, image_id):
        info = self.image_info[image_id]
        with open(info['yaml_path']) as f:
            temp = yaml.load(f.read(), Loader=yaml.FullLoader)
            labels = temp['label_names']
            del labels[0]
        return labels

    # 重新写draw_mask
    def draw_mask(self, num_obj, mask, image, image_id):
        info = self.image_info[image_id]
        for index in range(num_obj):
            for i in range(info['width']):
                for j in range(info['height']):
                    at_pixel = image.getpixel((i, j))
                    if at_pixel == index + 1:
                        mask[j, i, index] = 1
        return mask

    # 重新写load_shapes，里面包含自己的自己的类别
    #并在self.image_info信息中添加了path、mask_path 、yaml_path
    dataset_root_path = "/tongue_dateset/"
    yaml_pathdataset_root_path = "/tongue_dateset/"
    img_floder = dataset_root_path + "rgb"
    mask_floder = dataset_root_path + "mask"

    def load_shapes(self, count, img_floder, mask_floder, imglist, dataset_root_path):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        # Add classes
        self.add_class("shapes", 1, "word")  # 标线
        self.add_class("shapes", 2, "picture")  # 路面标志
        self.add_class("shapes", 3, "table")
        self.add_class("shapes", 4, "title")

        for i in range(val):
            i += train
            print(i)
            # 获取图片宽和高
            filestr = imglist[i].split(".")[0]
            # filestr = filestr.split("_")[1]
            mask_path = mask_floder + filestr + "_json/" + "label.png"
            #yaml_path = mask_floder + filestr + "_json/info.yaml"
            cv_img = cv2.imread(mask_floder + filestr + "_json/" + "img.png")
            print(img_floder + "/" + imglist[i])
            self.add_image("shapes", image_id=i, path=img_floder + "/" + imglist[i],
                           width=cv_img.shape[1], height=cv_img.shape[0], mask_path=mask_path, yaml_path=yaml_path)

    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        global iter_num
        info = self.image_info[image_id]
        count = 1  # number of object
        img = Image.open(info['mask_path'])
        num_obj = self.get_obj_index(img)
        mask = np.zeros([info['height'], info['width'], num_obj], dtype=np.uint8)
        mask = self.draw_mask(num_obj, mask, img, image_id)
        occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
        for i in range(count - 2, -1, -1):
            mask[:, :, i] = mask[:, :, i] * occlusion

            occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))
        labels = self.from_yaml_get_class(image_id)
        labels_form = []
        for i in range(len(labels)):
            if labels[i].find("word") != -1:
                labels_form.append("word")
            elif labels[i].find("picture") != -1:
                labels_form.append("picture")
            elif labels[i].find("table") != -1:
                labels_form.append("table")
            elif labels[i].find("title") != -1:
                labels_form.append("title")

        class_ids = np.array([self.class_names.index(s) for s in labels_form])
        return mask, class_ids.astype(np.int32)
# 测试集
class DrugDataset_test(utils.Dataset):
    # 得到该图中有多少个实例（物体）
    def get_obj_index(self, image):
        n = np.max(image)
        return n

    # 解析labelme中得到的yaml文件，从而得到mask每一层对应的实例标签
    def from_yaml_get_class(self, image_id):
        info = self.image_info[image_id]
        with open(info['yaml_path']) as f:
            temp = yaml.load(f.read(), Loader=yaml.FullLoader)
            labels = temp['label_names']
            del labels[0]
        return labels

    # 重新写draw_mask
    def draw_mask(self, num_obj, mask, image, image_id):
        info = self.image_info[image_id]
        for index in range(num_obj):
            for i in range(info['width']):
                for j in range(info['height']):
                    at_pixel = image.getpixel((i, j))
                    if at_pixel == index + 1:
                        mask[j, i, index] = 1
        return mask

    # 重新写load_shapes，里面包含自己的自己的类别
    # 并在self.image_info信息中添加了path、mask_path 、yaml_path
    dataset_root_path = "/tongue_dateset/"
    yaml_pathdataset_root_path = "/tongue_dateset/"
    img_floder = dataset_root_path + "rgb"
    mask_floder = dataset_root_path + "mask"

    def load_shapes(self, count, img_floder, mask_floder, imglist, dataset_root_path):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        # Add classes
        self.add_class("shapes", 1, "word")  # 标线
        self.add_class("shapes", 2, "picture")  # 路面标志
        self.add_class("shapes", 3, "table")
        self.add_class("shapes", 4, "title")

        for i in range(test):
            i += (train + val)
            print(i)
            # 获取图片宽和高
            filestr = imglist[i].split(".")[0]
            # filestr = filestr.split("_")[1]
            mask_path = mask_floder + filestr + "_json/" + "label.png"
            yaml_path = mask_floder + filestr + "_json/info.yaml"
            cv_img = cv2.imread(mask_floder + filestr + "_json/img.png")
            print(img_floder + "/" + imglist[i])
            self.add_image("shapes", image_id=i, path=img_floder + "/" + imglist[i],
                           width=cv_img.shape[1], height=cv_img.shape[0], mask_path=mask_path, yaml_path=yaml_path)

    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        global iter_num
        info = self.image_info[image_id]
        count = 1  # number of object
        img = Image.open(info['mask_path'])
        num_obj = self.get_obj_index(img)
        mask = np.zeros([info['height'], info['width'], num_obj], dtype=np.uint8)
        mask = self.draw_mask(num_obj, mask, img, image_id)
        occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
        for i in range(count - 2, -1, -1):
            mask[:, :, i] = mask[:, :, i] * occlusion

            occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))
        labels = self.from_yaml_get_class(image_id)
        labels_form = []  # class_id 必须与class_names对应
        for i in range(len(labels)):
            if labels[i].find("word") != -1:
                labels_form.append("word")
            elif labels[i].find("picture") != -1:
                labels_form.append("picture")
            elif labels[i].find("table") != -1:
                labels_form.append("table")
            elif labels[i].find("title") != -1:
                labels_form.append("title")

        class_ids = np.array([self.class_names.index(s) for s in labels_form])

        return mask, class_ids.astype(np.int32)


def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax


if __name__ == '__main__':
    # 基础设置
    dataset_root_path = r"E:\OCR_picture"
    img_floder = dataset_root_path + ""
    mask_floder = dataset_root_path + "label.png"
    imglist = os.listdir(img_floder)
    count = len(imglist)
    train = int(count * 0.85)
    val = int(count * 0.15)
    test = count - train - val
    # train与val数据集准备
    dataset_train = DrugDataset()
    dataset_train.load_shapes(train, img_floder, mask_floder, imglist, dataset_root_path)
    dataset_train.prepare()
    print('train_done')

    dataset_val = DrugDataset_val()
    dataset_val.load_shapes(val, img_floder, mask_floder, imglist, dataset_root_path)
    dataset_val.prepare()
    print('val_done')

    # dataset_test = DrugDataset_test()
    # dataset_test.load_shapes(test, img_floder, mask_floder, imglist, dataset_root_path)
    # dataset_test.prepare()
    # print('test_done')

    # Create model in training mode
    model = modellib.MaskRCNN(mode="training", config=config,
                              model_dir=MODEL_DIR)

    # Which weights to start with?
    init_with = "coco"  # imagenet, coco, or last

    if init_with == "imagenet":
        model.load_weights(model.get_imagenet_weights(), by_name=True)
    elif init_with == "coco":
        model.load_weights(COCO_MODEL_PATH, by_name=True,
                           exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                    "mrcnn_bbox", "mrcnn_mask"])
    elif init_with == "last":
        # Load the last model you trained and continue training
        model.load_weights(model.find_last(), by_name=True)

    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE / 20,
                epochs=100,
                layers="all")

