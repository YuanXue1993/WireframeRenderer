import argparse
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import numpy as np
import os
import pickle
from data.loaddata import get_loader
from models.model import Generator, NLayerDiscriminator
from models.sync_batchnorm import SynchronizedBatchNorm2d, DataParallelWithCallback
from models.MSSSIM import msssim
import torch.optim as optim
from torchvision import transforms
import torchvision.utils as vutils
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tensorboardX import SummaryWriter
from models.perceptual import PNet
from utils import checkpath

cudnn.benchmark = True

def main(args):
    # set which gpu(s) to use, should set PCI_BUS_ID first
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    num_gpus = (len(args.gpu) + 1)//2
    

    # create model directories
    checkpath(args.modelG_path)
    checkpath(args.modelD_path)

    # tensorboard writer
    checkpath(args.log_path)
    writer = SummaryWriter(args.log_path)

    # load data
    data_loader, num_train = get_loader(args, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, training=True)
    data_loader_val, num_test = get_loader(args, batch_size=args.val_bs, shuffle=False, num_workers=args.num_workers, training=False)
    print('Finished data loading')
    print("The length of the train set is: {}".format(num_train))
    print("The length of the test set is: {}".format(num_test))

    colorguide = True
    if args.nocolor:
        colorguide = False
    
    # loss multipliers
    lambdas = [args.lambda_imgl1, args.lambda_wfl1, args.lambda_ssim, args.lambda_color]
    lambda_perceptual = args.lambda_perceptual

    # Generator
    netG = Generator(lambdas=lambdas, colorguide=colorguide, input_nc=1, output_nc=1)

    if num_gpus > 1:
        # multi-gpu training with synchonized batchnormalization
        # make sure enough number of gpus are available
        assert(torch.cuda.device_count() >= num_gpus)
        # since we have set CUDA_VISIBLE_DEVICES to avoid some invalid device id issues
        netG = DataParallelWithCallback(netG, device_ids=[i for i in range(num_gpus)])
        netG_single = netG.module
    else:
        # single gpu training
        netG_single = netG

    # Discriminator
    netD = NLayerDiscriminator(input_nc=4, n_layers=4)
    if num_gpus > 1:
        netD = DataParallelWithCallback(netD, device_ids=[i for i in range(num_gpus)])
        netD_single = netD.module
    else:
        netD_single = netD

    # print(netG_single)
    # print(netD_single)

    if args.pretrained and args.netG_path != '' and args.netD_path != '':
        netG_single.load_state_dict(torch.load(args.netG_path))
        netD_single.load_state_dict(torch.load(args.netD_path))

    # Right now we only support gpu training
    if torch.cuda.is_available():
        netG = netG.cuda()
        netD = netD.cuda()

    # define the perceptual loss, place outside the forward func in G for better multi-gpu training
    Ploss = PNet()
    if num_gpus > 1:
        Ploss = DataParallelWithCallback(Ploss, device_ids=[i for i in range(num_gpus)])

    if torch.cuda.is_available():
        Ploss = Ploss.cuda()
    
    # setup optimizer
    lr = args.learning_rate
    optimizerD = optim.Adam(netD_single.parameters(), lr=lr, betas=(args.beta1, 0.999))
    schedulerD = ReduceLROnPlateau(optimizerD, factor = 0.7, patience = 10, mode = 'min', min_lr = 1e-06)
    optimizerG = optim.Adam(netG_single.parameters(), lr=lr, betas=(args.beta1, 0.999))
    schedulerG = ReduceLROnPlateau(optimizerG, factor = 0.7, patience = 10, mode = 'min', min_lr = 1e-06)



    for epoch in range(args.num_epochs):
        # switch to train mode
        netG.train()
        netD.train()

        for i, (img_real, wf_real, color_real) in enumerate(data_loader, 0):
            img_real = img_real.cuda()
            wf_real = wf_real.cuda()
            color_real = color_real.cuda()

            # Update D network, we freeze parameters in G to save memory
            for p in netG_single.parameters():
                p.requires_grad=False
            for p in netD_single.parameters():
                p.requires_grad=True

            # if using TTUR, D can be trained multiple steps per G step    
            for _ in range(args.D_steps):
                optimizerD.zero_grad()
                
                # train with real
                real_AB = torch.cat((img_real, wf_real), 1)
                errD_real = 0.5 * netD(trainG=False, trainReal=True, real_AB=real_AB, fake_AB=None).sum()
                errD_real.backward()

                # train with fake
                img_fake, wf_fake, _, _, _, _, _ = netG(trainG=False, img_real=None, wf_real=wf_real, color_real=color_real)
                fake_AB = torch.cat((img_fake, wf_fake), 1)
                errD_fake = 0.5 * netD(trainG=False, trainReal=False, real_AB=None, fake_AB=fake_AB).sum()
                errD_fake.backward()             

                errD = errD_real + errD_fake
                optimizerD.step()
                del img_fake, wf_fake, fake_AB, real_AB, errD_real, errD_fake

            iterations_before_epoch = epoch*len(data_loader)
            writer.add_scalar('D Loss', errD.item(), iterations_before_epoch + i)
            del errD

            # Update G network, we freeze parameters in D to save memory
            for p in netG.parameters():
                p.requires_grad=True
            for p in netD.parameters():
                p.requires_grad=False
            
            optimizerG.zero_grad()

            img_fake, wf_fake, lossG, wf_ssim, img_l1, color_l1, wf_l1 = netG(trainG=True, img_real=img_real, wf_real=wf_real, color_real=color_real)
            ploss = Ploss(img_fake, img_real.detach()).sum()         
            fake_AB = torch.cat((img_fake, wf_fake), 1)
            lossD = netD(trainG=True, trainReal=False, real_AB=None, fake_AB=fake_AB).sum()            
            errG = (lossG.sum() + lambda_perceptual * ploss + lossD)
            errG.backward()
            optimizerG.step()

            del color_real, fake_AB, lossG, errG

            if args.nocolor:
                print('Epoch: [{}/{}] Iter: [{}/{}] PercLoss : {:.4f} ImageL1 : {:.6f} WfL1 : {:.6f} WfSSIM : {:.6f}'
                    .format(epoch, args.num_epochs, i, len(data_loader), ploss.item(), img_l1.sum().item(), wf_l1.sum().item(), num_gpus + wf_ssim.sum().item()))
            else:
                print('Epoch: [{}/{}] Iter: [{}/{}] PercLoss : {:.4f} ImageL1 : {:.6f} WfL1 : {:.6f} WfSSIM : {:.6f} ColorL1 : {:.6f}'
                    .format(epoch, args.num_epochs, i, len(data_loader), ploss.item(), img_l1.sum().item(), wf_l1.sum().item(), num_gpus + wf_ssim.sum().item(), color_l1.sum().item()))
                writer.add_scalar('Color Loss', color_l1.sum().item(), iterations_before_epoch + i)
            
            # tensorboard log
            writer.add_scalar('G Loss', lossD.item(), iterations_before_epoch + i)
            writer.add_scalar('Image L1 Loss', img_l1.sum().item(), iterations_before_epoch + i)
            writer.add_scalar('Wireframe MSSSIM Loss', num_gpus + wf_ssim.sum().item(), iterations_before_epoch + i)
            writer.add_scalar('Wireframe L1', wf_l1.sum().item(), iterations_before_epoch + i)
            writer.add_scalar('Image Perceptual Loss', ploss.item(), iterations_before_epoch + i)

            del wf_ssim, ploss, img_l1, color_l1, wf_l1, lossD

            with torch.no_grad():
                # show generated tarining images in tensorboard
                if i % args.val_freq == 0:
                    real_img = vutils.make_grid(img_real.detach()[:args.val_size], normalize=True, scale_each=True)
                    writer.add_image('Real Image', real_img, (iterations_before_epoch + i)//args.val_freq)
                    real_wf= vutils.make_grid(wf_real.detach()[:args.val_size], normalize=True, scale_each=True)
                    writer.add_image('Real Wireframe', real_wf, (iterations_before_epoch + i)//args.val_freq)
                    fake_img = vutils.make_grid(img_fake.detach()[:args.val_size], normalize=True, scale_each=True)
                    writer.add_image('Fake Image', fake_img, (iterations_before_epoch + i)//args.val_freq)
                    fake_wf = vutils.make_grid(wf_fake.detach()[:args.val_size], normalize=True, scale_each=True)
                    writer.add_image('Fake Wireframe', fake_wf, (iterations_before_epoch + i)//args.val_freq)
                    del real_img, real_wf, fake_img, fake_wf

            del img_real, wf_real, img_fake, wf_fake

        # do checkpointing
        if epoch % args.save_freq == 0 and epoch > 0:
            torch.save(netG_single.state_dict(), '{}/netG_epoch_{}.pth'.format(args.modelG_path, epoch))
            torch.save(netD_single.state_dict(), '{}/netD_epoch_{}.pth'.format(args.modelD_path, epoch))

        # validation
        with torch.no_grad():
            netG_single.eval()
            # since we use a realtively large validation batchsize, we don't go through the who test set
            (img_real, wf_real, color_real) = next(iter(data_loader_val))
            img_real = img_real.cuda()
            wf_real = wf_real.cuda()
            color_real = color_real.cuda()

            img_fake, wf_fake, _, _, _, _, _ = netG_single(trainG=False, img_real=None, wf_real=wf_real, color_real=color_real)

            # update lr based on the validation perceptual loss
            val_score = Ploss(img_fake.detach(), img_real.detach()).sum()
            schedulerG.step(val_score)
            schedulerD.step(val_score)
            print('Current lr: {:.6f}'.format(optimizerG.param_groups[0]['lr']))


            real_img = vutils.make_grid(img_real.detach()[:args.val_size], normalize=True, scale_each=True)
            writer.add_image('Test: Real Image', real_img, epoch)
            real_wf = vutils.make_grid(wf_real.detach()[:args.val_size], normalize=True, scale_each=True)
            writer.add_image('Test: Real Wireframe', real_wf, epoch)
            fake_img = vutils.make_grid(img_fake.detach()[:args.val_size], normalize=True, scale_each=True)
            writer.add_image('Test: Fake Image', fake_img, epoch)
            fake_wf = vutils.make_grid(wf_fake.detach()[:args.val_size], normalize=True, scale_each=True)
            writer.add_image('Test: Fake Wireframe', fake_wf, epoch)

            netG_single.train()

            del img_real, real_img, wf_real, real_wf, img_fake, fake_img, wf_fake, fake_wf

    # close tb writer
    writer.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--root_path', type=str, default='./data',
                        help='root path for wireframe dataset')
    parser.add_argument('--log_path', type=str, default='./results/tb_logs/wfrenderer',
                        help='path for saving tensorboard logs')
    parser.add_argument('--modelG_path', type=str, default='./results/saved_models/wfrenderer_G',
                        help='path for saving trained G models')
    parser.add_argument('--modelD_path', type=str, default='./results/saved_models/wfrenderer_D',
                        help='path for saving trained G models')
    parser.add_argument('--netG_path', type=str, default='',
                        help='path for loading saved G models (to continue training)')
    parser.add_argument('--netD_path', type=str, default='',
                        help="path for loading saved D models (to continue training)")
    parser.add_argument('--nocolor', action='store_true',
                        help='not using color guided model, needs to be also specified when testing the model')
    parser.add_argument('--D_steps', type=int, default=1,
                        help='number of training D steps for TTUR')
    parser.add_argument('--lambda_imgl1', type=int, default=1)
    parser.add_argument('--lambda_wfl1', type=int, default=1)
    parser.add_argument('--lambda_ssim', type=int, default=1)
    parser.add_argument('--lambda_perceptual', type=int, default=10)
    parser.add_argument('--lambda_color', type=int, default=10)                    
    parser.add_argument('--val_freq', type=int, default=20)
    parser.add_argument('--save_freq', type=int, default=5)                    
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=40)
    parser.add_argument('--val_bs', type=int, default=60)
    parser.add_argument('--val_size', type=int, default=40,
                        help='# of iamges shown in tensorboard')
    parser.add_argument('--beta1', type=float, default=0.5,
                        help='beta1 for adam. default=0.5')
    parser.add_argument('--img_size', type=int, default=256,
                        help='default image size for the wireframe renderer')
    parser.add_argument('--learning_rate', type=float, default=0.0002)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--gpu', type=str, default='0,1,2,3')
    
    args = parser.parse_args()
    print(args)
    main(args)