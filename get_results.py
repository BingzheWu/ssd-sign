#!/usr/bin/env python
# coding=utf-8
import sys
import mxnet as mx
from detect.detector  import Detector
import argparse
import importlib
import cPickle
import os
CLASSES = cPickle.load(open('sign_classes', 'rb'))
def get_img_list(test_file, sign_root = '/opt/dataset/sign/DataSet'):
    img_list = []
    with open(test_file, 'r') as f:
        for img_file in f.readlines():
            image_class, _, image_path = img_file.split(' ')
            if image_class!='no-logo':
                image_file_path = os.path.join(sign_root, image_path.strip())
                img_list.append(image_file_path)
    return img_list
def parse_args():
    parser = argparse.ArgumentParser(description='Single-shot detection network demo')
    parser.add_argument('--network', dest='network', type=str, default='vgg16_reduced',
                        choices=['vgg16_reduced'], help='which network to use')
    parser.add_argument('--images', dest='images', type=str, default='./data/apple.jpeg',
                        help='run demo with images, use comma(without space) to seperate multiple images')
    parser.add_argument('--dir', dest='dir', nargs='?',
                        help='demo image directory, optional', type=str)
    parser.add_argument('--ext', dest='extension', help='image extension, optional',
                        type=str, nargs='?')
    parser.add_argument('--epoch', dest='epoch', help='epoch of trained model',
                        default=0, type=int)
    parser.add_argument('--prefix', dest='prefix', help='trained model prefix',
                        default=os.path.join(os.getcwd(), 'model', 'ssd'), type=str)
    parser.add_argument('--cpu', dest='cpu', help='(override GPU) use CPU to detect',
                        action='store_true', default=False)
    parser.add_argument('--gpu', dest='gpu_id', type=int, default=0,
                        help='GPU device id to detect with')
    parser.add_argument('--data-shape', dest='data_shape', type=int, default=300,
                        help='set image shape')
    parser.add_argument('--mean-r', dest='mean_r', type=float, default=123,
                        help='red mean value')
    parser.add_argument('--mean-g', dest='mean_g', type=float, default=117,
                        help='green mean value')
    parser.add_argument('--mean-b', dest='mean_b', type=float, default=104,
                        help='blue mean value')
    parser.add_argument('--thresh', dest='thresh', type=float, default=0.6,
                        help='object visualize score threshold, default 0.6')
    parser.add_argument('--nms', dest='nms_thresh', type=float, default=0.5,
                        help='non-maximum suppression threshold, default 0.5')
    parser.add_argument('--force', dest='force_nms', type=bool, default=True,
                        help='force non-maximum suppression on different class')
    parser.add_argument('--timer', dest='show_timer', type=bool, default=True,
                        help='show detection time')
    args = parser.parse_args()
    return args

def get_detector(net, prefix, epoch, data_shape, mean_pixels, ctx,
                 nms_thresh = 0.5, force_nms= True):
    sys.path.append(os.path.join(os.getcwd(), 'symbol'))
    net = importlib.import_module("symbol_"+net)\
            .get_symbol(len(CLASSES), nms_thresh, force_nms)
    detector = Detector(net, prefix + "_"+ str(data_shape), epoch, \
                       data_shape, mean_pixels, ctx = ctx)
    return detector
def get_det_results(test_file):
    args = parse_args()
    if args.cpu:
        ctx = mx.cpu()
    else:
        ctx = mx.gpu(args.gpu_id)
    img_list = get_img_list(test_file)
    detector = get_detector(args.network, args.prefix, args.epoch,
                                                       args.data_shape,
                                                       (args.mean_r, args.mean_g, args.mean_b),
                                                       ctx, args.nms_thresh, args.force_nms)
    #results = detector.im_detect(img_list, show_timer = True)
    #cPickle.dump(results, open('dets','wb'))
    results = cPickle.load(open('dets', 'rb'))
    for result in results:
        print(result)
if __name__ =='__main__':
    get_det_results(test_file = '/opt/dataset/sign/DataSet/testset.txt')

    


