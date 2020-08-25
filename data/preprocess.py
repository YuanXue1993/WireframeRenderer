import os
import argparse
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import cv2
import numpy as np
import pickle
from glob import glob
import sys
sys.path.append(".")
from utils import intx, coord, checkpath


def main(args):
    # Create output directory
    checkpath(args.out_path)

    # list all files end with jpg, args.in_path should contain a folder with name imgs for jpg files
    # and a folder with name pointlines for pkl files.
    input_paths_train = sorted(glob(os.path.join(args.in_path, '{}/*.jpg'.format("imgs/train"))))
    input_paths_test = sorted(glob(os.path.join(args.in_path, '{}/*.jpg'.format("imgs/test"))))
    if len(input_paths_train) == 0 or len(input_paths_test) == 0:
        raise Exception("No images are found in {}".format(args.in_path))

    # filter out outdoor images labeled by us, this is optinal
    outdoor_list_train = os.path.join(args.in_path, 'outdoor_list_train.txt')
    outdoor_list_test = os.path.join(args.in_path, 'outdoor_list_test.txt')

    with open(outdoor_list_train,'r') as fp:
	    outdoor_train = fp.readlines()
    # the last one doesn't contain the '\n' so processed seperately
    outdoor_names_train = [ p[:-2] for p in outdoor_train if len(p) > 9]
    outdoor_names_train += [outdoor_train[-1][:-1]]
    print(len(outdoor_names_train))

    with open(outdoor_list_test,'r') as fp:
	    outdoor_test = fp.readlines()
    outdoor_names_test = [ p[:-2] for p in outdoor_test if len(p) > 9]
    outdoor_names_test += [outdoor_test[-1][:-1]]

    data = []
    data_train = []
    data_test = []

    # save training data
    for fname in input_paths_train:
        # retrive the index name of the image
        basename = fname.split('/')[-1].split('.')[0]
        if basename not in outdoor_names_train:
            filename = os.path.join(args.in_path, 'pointlines/{}.pkl'.format(basename))
            _ = process(args.uni_wf, args.img_size, args.out_path, filename, mode='Train')
            item = (filename, 'Train')
            data_train.append(item)
            data.append(item)

    # save test data
    for fname in input_paths_test:
        # retrieve the index name of the image
        basename = fname.split('/')[-1].split('.')[0]
        if basename not in outdoor_names_test:
            filename = os.path.join(args.in_path, 'pointlines/{}.pkl'.format(basename))
            _ = process(args.uni_wf, args.img_size, args.out_path, filename, mode='Test')
            item = (filename, 'Test')
            data_test.append(item)
            data.append(item)

    print("The length of the dataset is: {}".format(len(data)))
    print("The length of the training set is: {}".format(len(data_train)))
    print("The length of the test set is: {}".format(len(data_test)))
    print('finished preprocessing data')

# get img-wf pairs
def process(uni_wf, out_size, out_path, filename, mode):
    # get output paths for preprocessed imgs and wireframes
    basename = filename.split('/')[-1].split('.')[0]
    wf_dir_train = os.path.join(out_path, 'wireframes/train')
    checkpath(wf_dir_train)
    img_dir_train = os.path.join(out_path, 'images/train')
    checkpath(img_dir_train)

    wf_dir_test = os.path.join(out_path, 'wireframes/test')
    checkpath(wf_dir_test)    
    img_dir_test = os.path.join(out_path, 'images/test')
    checkpath(img_dir_test)

    with open(filename, 'rb') as f:
        target = pickle.load(f, encoding='latin1')
        img = target['img']
        h, w, _ = img.shape
        img_size = np.array((w, h))
        img = cv2.resize(img, (out_size, out_size), interpolation=cv2.INTER_AREA)
        if mode == 'Train':
            img_dir = os.path.join(img_dir_train, '{}.png'.format(basename))
        elif mode == 'Test':
            img_dir = os.path.join(img_dir_test, '{}.png'.format(basename))
        cv2.imwrite(img_dir, img)

        points = target['points']
        lines = target['lines']
        wf = np.zeros((out_size, out_size))

        for i, j in lines:
            start = np.array( points[i] ) * out_size / img_size
            end = np.array( points[j] ) * out_size / img_size
            if uni_wf:
                # use unified intensity
                dist = 1
            else:
                # different intensity represents different dists, optional and haven't been thoroughly tested
                dist = np.linalg.norm(end - start) / (out_size * np.sqrt(2))
                if dist < 0.1:
                    dist = 0.2
                elif dist > 0.5:
                    dist = 1
                else:
                    dist = dist * 2
            # haven't experimented with antialiased lines, user can optinally try lineType=cv2.LINE_AA
            wf = cv2.line(wf, intx(start, out_size), intx(end, out_size), 255 * dist, 1, lineType=cv2.LINE_8)

        if mode == 'Train':
            save_dir = os.path.join(wf_dir_train, '{}.png'.format(basename))
        elif mode == 'Test':
            save_dir = os.path.join(wf_dir_test, '{}.png'.format(basename))

        cv2.imwrite(save_dir, wf)

    return wf


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_path', type=str, default='./data/raw_data',
                        help='input path containing raw data for wireframe dataset')
    parser.add_argument('--out_path', type=str, default='./data',
                        help='path for saving processed data')
    parser.add_argument('--img_size', type=int, default=256,
                        help='default image size for the wireframe renderer')
    parser.add_argument('--uni_wf', action='store_true',
                        help='use unified intensity in rasterized wireframes, this is the default setting')

    args = parser.parse_args()
    print(args)
    main(args)
