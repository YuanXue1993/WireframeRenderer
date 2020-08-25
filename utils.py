from PIL import Image
import os
import numpy as np
from scipy import linalg
import math
from math import exp
import torch

def intx(x, out_size):
    return(min(math.floor(x[0]), out_size-1), min(math.floor(x[1]), out_size-1))

def coord(x, out_size):
    return(min(math.floor(x[1]), out_size-1), min(math.floor(x[0]), out_size-1))

def has_file_allowed_extension(filename, extensions):
    return any(filename.endswith(ext) for ext in extensions)

def checkpath(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as e:
            if e.errno != os.errno.EEXIST:
                raise   
            # time.sleep might help here
            pass
    return path

def default_loader_img(filename):
    img = Image.open(filename).convert('RGB')
    return img

def default_loader_wf(filename):
    wf = Image.open(filename).convert('L')
    return wf

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def save_singleimages(images, folder, startID, imsize):
    for i in range(images.size(0)):
        fullpath = '%s/%d_%d.png' % (folder, startID + i, imsize)
        # range from [-1, 1] to [0, 255]
        img = images[i].add(1).div(2).mul(255).clamp(0, 255).byte()
        ndarr = img.permute(1, 2, 0).data.cpu().numpy()
        im = Image.fromarray(ndarr)
        im.save(fullpath)

def compute_fid_score(preds1, preds2, eps=1e-6):
    mu1 = np.mean(preds1, axis=0)
    sigma1 = np.cov(preds1, rowvar=False)

    mu2 = np.mean(preds2, axis=0)
    sigma2 = np.cov(preds2, rowvar=False)

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)

    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)