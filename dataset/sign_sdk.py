#!/usr/bin/env python
# coding=utf-8
from scipy import misc
import os
import numpy as np
from PIL import Image
from imdb import Imdb
import cv2
IMG_EXTENSIONS = [
        '.jpg', '.JPG', '.jpeg', '.JPEG',
        '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
        return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)
def default_image_loader(path):
    img = cv2.imread(path)
    #img = misc.imread(path)
    img = img.transpose((2,0,1))
    return img

class sign(Imdb):
    def __init__(self, sign_root, loader = default_image_loader, shuffle = False, is_train = True):
        super(sign, self).__init__('sign')
        self.sign_root = sign_root
        self.classes = self.get_classes()
        self.num_classes = len(self.classes)
        self.is_train = is_train
        self.class_idx = self.class2idx()
        self.image_set_index, self.annotations = self.make_dataset()
        self.num_images = len(self.image_set_index)
        self.loader = loader
        if self.is_train:
            self.labels = self._load_image_labels()
    def get_classes(self, mode = 'All'):
        if mode == 'All':
            traffic_sign = os.path.join(self.sign_root, 'Images', mode)
            classes = os.listdir(traffic_sign)
        return classes 
    def class2idx(self):
        class_idx = {self.classes[i]:  i for i in range(len(self.classes))}
        return class_idx
    def make_dataset(self, mode = 'All'):
        images = []
        annotations = []
        images_dir = os.path.join(self.sign_root, 'Images', mode)
        annotations_dir = os.path.join(self.sign_root, 'Annotations', mode)
        for target in self.classes:
            d = os.path.join(annotations_dir, target)
            if not os.path.isdir(d):
                continue
            for root, _, fnames in sorted(os.walk(d)):
                for fname in fnames:
                    path = os.path.join(root, fname)
                    image_path = path.split('/')[-1].split('.bboxes.txt')[0]
                    image_path = os.path.join(self.sign_root, 'Images', 'All', target, image_path)
                    item = (path, self.class_idx[target])
                    images.append(image_path.strip())
                    annotations.append(item)
        return images, annotations
    def image_path_from_index(self, index):
        name = self.image_set_index[index]
        assert os.path.exists(name)
        return name
    def label_from_index(self, index):
        return self.labels[index]
    def _label_path_from_index(self, index):
        label_file = self.annotations[index][0]
        return label_file
    def _load_image_labels(self):
        '''
        returns:
        labels packed in [num_images*max_num_objects*5] tensor
        '''
        temp = []
        for index in range(self.num_images):
            label = []
            difficult = 1.5
            label_file = self.annotations[index][0]
            image_file = self.image_set_index[index]
            cls_id = self.annotations[index][1]
            xmin, ymin, xmax, ymax = self.load_boxes(label_file)
            c,h,w = self.loader(image_file).shape
            xmin = float(xmin) / w
            ymin = float(ymin) / h
            xmax = float(xmax) / w
            ymax = float(ymax) / h
            label.append([cls_id, xmin, ymin, xmax, ymax, difficult])
            temp.append(np.array(label))
        return temp
    def load_boxes(self, box_path):
        with open(box_path, 'r') as r:
            boxes =[ float(i) for i in r.readlines()[1].strip().split(' ')]
        xmin = boxes[0] 
        ymin = boxes[1]
        xmax = boxes[0] + boxes[2] 
        ymax = boxes[1] + boxes[3]
        return xmin, ymin, xmax, ymax
def imshow(img, box):
    import matplotlib.pyplot as plt
    x0 = box[0] -  box[2] /2
    x0 = box[0]
    y0 = box[1] - box[3] / 2
    y0 = box[1]
    x1 = x0+box[2]
    y1 = y0+box[3]
    cv2.rectangle(img, (x0, y0), (x1, y1), color = (0, 255, 0))
    cv2.imshow('image', img)
    cv2.waitKey(-1)
    plt.show()
if __name__ == '__main__':
    sign_ = sign('/opt/dataset/sign/DataSet')
