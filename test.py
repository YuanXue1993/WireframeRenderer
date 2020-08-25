import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np
import os
from data.loaddata import get_loader
from models.model import Generator, NLayerDiscriminator
from models.MSSSIM import ssim, msssim
from models.perceptual import PNet
from models.inceptionv3 import InceptionV3
from utils import save_singleimages, checkpath, compute_fid_score

cudnn.benchmark = True


def main(args):
    # by default we only consider single gpu inference
    assert(len(args.gpu) == 1)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # load data
    data_loader_val, num_test = get_loader(args, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, training=False)
    print('finished data loading')

    # Generator
    colorguide = True
    if args.nocolor:
        colorguide = False
    netG = Generator(lambdas=None, colorguide=colorguide, input_nc=1, output_nc=1)

    netG.load_state_dict(torch.load(args.model_path))

    if torch.cuda.is_available():
        netG = netG.cuda()

    out_path = args.out_path
    checkpath(out_path)

    predictions_fid_real = []
    predictions_fid_fake = []
    fid_model = InceptionV3().cuda()
    fid_model.eval()
    Perceptual = PNet().cuda()

    avg_ssim = 0
    lpips = 0

    # validate on test set, TODO: test with single color guide image
    with torch.no_grad():
        netG.eval()
        for i, (img_real, wf_real, color_real) in enumerate(data_loader_val, 0):
            img_real = img_real.cuda()
            wf_real = wf_real.cuda()
            if colorguide:
                color_real = color_real.cuda()
            # in case we are in the last interation
            batch_size = img_real.size(0)

            img_fake, wf_fake, _, _, _, _, _ = netG(trainG=False, img_real=None, wf_real=wf_real, color_real=color_real)

            ssim_score = ssim(img_real, img_fake).item() * batch_size
            avg_ssim += ssim_score

            lpips += Perceptual(img_real, img_fake) * batch_size

            # TODO: save generated wireframes
            save_singleimages(img_fake, out_path, i*args.batch_size, args.img_size)

            pred_fid_real = fid_model(img_real)[0]
            pred_fid_fake = fid_model(img_fake)[0]
            predictions_fid_real.append(pred_fid_real.data.cpu().numpy().reshape(batch_size, -1))
            predictions_fid_fake.append(pred_fid_fake.data.cpu().numpy().reshape(batch_size, -1))

        print('SSIM: {:6f}'.format(avg_ssim/num_test))

        print('LPIPS: {:6f}'.format(lpips/num_test))

        predictions_fid_real = np.concatenate(predictions_fid_real, 0)
        predictions_fid_fake = np.concatenate(predictions_fid_fake, 0)
        fid = compute_fid_score(predictions_fid_fake, predictions_fid_real)
        print('FID: {:6f}'.format(fid))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nocolor', action='store_true',
                        help='not using color guided model, needs to be also specified when training the model')
    parser.add_argument('--model_path', type=str, default='./results/saved_models/wfrenderer_G/netG_epoch_300.pth',
                        help='path for saved G model')
    parser.add_argument('--root_path', type=str, default='./data',
                        help='root path for wireframe dataset')
    parser.add_argument('--out_path', type=str, default='./results/out_imgs',
                        help='path for saving rendered images')
    parser.add_argument('--img_size', type=int, default=256,
                        help='default image size for the wireframe renderer')
    parser.add_argument('--batch_size', type=int, default=40)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--gpu', type=str, default='0')
    args = parser.parse_args()
    print(args)
    main(args)