import os
import argparse
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.utils.data as data
from PIL import Image
import cv2
import numpy as np
from glob import glob
import random
import sys
sys.path.append(".")
from utils import default_loader_img, default_loader_wf

def make_dataset(args, dir, training):
    # list img files end with png
    if training:
        img_paths = sorted(glob(os.path.join(dir, '{}/*.png'.format("images/train"))))
        wf_paths = [p.replace('images/train', 'wireframes/train') for p in img_paths]
        # print("The length of the training set is: {}".format(len(img_paths)))
    else:
        img_paths = sorted(glob(os.path.join(dir, '{}/*.png'.format("images/test"))))
        wf_paths = [p.replace('images/test', 'wireframes/test') for p in img_paths]
        # print("The length of the test set is: {}".format(len(img_paths)))

    # return img-wf pairs
    return img_paths, wf_paths

def custom_transform(img, wf, size):
    if random.random() < 0.5:
        # random crop for both img/wf
        new_size = int(size*1.2)
        # different interpolations can be used here, haven't thoroughly tested
        img = transforms.Resize((new_size, new_size), Image.LANCZOS)(img)
        wf = transforms.Resize((new_size, new_size), Image.LANCZOS)(wf)
        i = random.randint(0, new_size - size)
        j = random.randint(0, new_size - size)
        img = img.crop((i, j, i + size, j + size))
        wf = wf.crop((i, j, i + size, j + size))
    else:
        w, h = img.size
        if h != w or h != size:
            img = transforms.Resize((size, size), Image.LANCZOS)(img)
            wf = transforms.Resize((size, size), Image.LANCZOS)(wf)

    # optional color jitter augmentation
    # img = transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0)(img)

    # use the same seed to control random horizontal flip for both img and wf
    seed = np.random.randint(123321)
    random.seed(seed)
    img = transforms.RandomHorizontalFlip(p=0.5)(img)
    random.seed(seed)
    wf = transforms.RandomHorizontalFlip(p=0.5)(wf)

    color_histogram = img.histogram() # used in color guided rendering
    color_histogram = torch.tensor(color_histogram, dtype=torch.float)/float(size*size)
    img = transforms.ToTensor()(img)
    wf = transforms.ToTensor()(wf)

    # conventional normalization for gan models
    img = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(img)
    wf = transforms.Normalize(mean=[0.5], std=[0.5])(wf)

    return img, wf, color_histogram

def custom_transform_eval(img, wf, size):
    w, h = img.size
    if h != w or h != size:
        img = transforms.Resize((size, size), Image.LANCZOS)(img)
        wf = transforms.Resize((size, size), Image.LANCZOS)(wf)

    color_histogram = img.histogram()
    color_histogram = torch.tensor(color_histogram, dtype=torch.float)/float(size*size)

    img = transforms.ToTensor()(img)
    wf = transforms.ToTensor()(wf)

    img = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(img)
    wf = transforms.Normalize(mean=[0.5], std=[0.5])(wf)
    
    return img, wf, color_histogram  

def get_loader(args, batch_size, shuffle=True, num_workers=16, training=True):
    """Returns torch.utils.data.DataLoader for wireframe dataset."""
    dataset = WireframeDataset(args, training=training)
    num_imgs = len(dataset)
    if training:
        data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  num_workers=num_workers,
                                                  drop_last=True)
    else:
        data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  num_workers=num_workers,
                                                  drop_last=False)
    return data_loader, num_imgs

class WireframeDataset(data.Dataset):
    """Wireframe Dataset compatible with torch.utils.data.DataLoader."""
    def __init__(self, args, training):

        self.args = args
        self.out_size = args.img_size
        self.training = training

        self.root = args.root_path
        if not os.path.exists(self.root):
            raise Exception("[!] {} not exists.".format(self.root))

        # return paths and labels
        samples_image, samples_wf = make_dataset(self.args, self.root, training=self.training)

        self.images = samples_image
        self.wireframes = samples_wf

    def __getitem__(self, index):
        """Returns (augumented) wireframe data."""
        # retrieve the img-line pairs
        img_path = self.images[index]
        wf_path = self.wireframes[index]
        img = default_loader_img(img_path)
        wf = default_loader_wf(wf_path)

        if self.training:
            img, wf, color_histogram = custom_transform(img, wf, size=self.out_size)
        else:
            img, wf, color_histogram = custom_transform_eval(img, wf, size=self.out_size)

        return img, wf, color_histogram

    def __repr__(self):
        fmt_str = 'Dataset: Wireframes' + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        return fmt_str


    def __len__(self):
        return len(self.images)
