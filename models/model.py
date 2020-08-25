""" partial code adapted from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix """
import torch
import torch.nn as nn
import torch.nn.init as init
from .sync_batchnorm import SynchronizedBatchNorm1d, SynchronizedBatchNorm2d, DataParallelWithCallback
from .MSSSIM import msssim

class Generator(nn.Module):
    # TODO: use vectorized wireframes as input instead of rasterized wireframes
    def __init__(self, lambdas, colorguide, input_nc, output_nc, ngf=64, norm_layer=SynchronizedBatchNorm2d, use_dropout=False, n_blocks=4, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(Generator, self).__init__()
        self.lambdas = lambdas
        self.colorguide = colorguide

        if padding_type =='reflect':
            padding = nn.ReflectionPad2d
        elif padding_type == 'replicate':
            padding = nn.ReplicationPad2d

        if colorguide:
            model = [padding(3),
                    # +3 is for color encoding(rgb)
                    nn.Conv2d(input_nc+3, ngf, kernel_size=7, stride=1, padding=0, bias=False),
                    norm_layer(ngf),
                    nn.LeakyReLU(0.2, True)]
        else:
            model = [padding(3),
                    nn.Conv2d(input_nc, ngf, kernel_size=7, stride=1, padding=0, bias=False),
                    norm_layer(ngf),
                    nn.LeakyReLU(0.2, True)]

        mult = 1
        mult_new = mult * 2
        for i in range(n_blocks):  # add downsampling layers
            model += [padding(1),
                      nn.Conv2d(ngf * mult, ngf * mult_new, kernel_size=3, stride=2, padding=0, bias=False),
                      norm_layer(ngf * mult_new),
                      nn.LeakyReLU(0.2, True)]
            mult = 2 * mult
            mult_new = mult * 2

        # a sequence of residual blocks
        for i in range(n_blocks):              
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=False)]


        if colorguide:
            # 3 X 256 color channels for rgb
            colortrans = [nn.Linear(3, 3 * 256, bias=False),
                        SynchronizedBatchNorm1d(256),
                        nn.LeakyReLU(0.2, True)]
            self.enc_color = nn.Sequential(*colortrans)
            del colortrans
        self.enc = nn.Sequential(*model)
        del model

        model_img = []
        model_wf = []
        for i in range(n_blocks):  # add upsampling layers
            mult = 2 ** (n_blocks - i)
            model_img += [padding(1),
                      nn.Conv2d(ngf * mult, int(ngf * mult * 2),
                                         kernel_size=3, stride=1,
                                         padding=0,
                                         bias=False),
                      nn.PixelShuffle(2),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]

            model_wf += [padding(1),
                      nn.Conv2d(ngf * mult, int(ngf * mult * 2),
                                         kernel_size=3, stride=1,
                                         padding=0,
                                         bias=False),
                      nn.PixelShuffle(2),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]

        self.dec_base_wf = nn.Sequential(*model_wf)
        self.dec_base_img = nn.Sequential(*model_img)

        model_img = [padding(3)]
        model_img += [nn.Conv2d(int(ngf * mult), output_nc*3, kernel_size=7, padding=0)]
        model_img += [nn.Tanh()]       

        model_wf = [padding(3)]
        model_wf += [nn.Conv2d(int(ngf * mult / 2), output_nc, kernel_size=7, padding=0)]
        model_wf += [nn.Tanh()]

        self.dec_img = nn.Sequential(*model_img)
        self.dec_wf = nn.Sequential(*model_wf)
        del model_img, model_wf

        if colorguide:
            model_color = [nn.Linear(256*3, 3, bias=False),
                        nn.Sigmoid()]
            self.dec_color = nn.Sequential(*model_color)
            del model_color


        # weight init
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv2d') != -1:
                init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    init.normal_(m.bias.data)
            elif classname.find('BatchNorm2d') != -1:
                init.normal_(m.weight.data, mean=1, std=0.02)
                if m.bias is not None:
                    init.constant_(m.bias.data, 0)
            elif classname.find('BatchNorm1d') != -1:
                init.normal_(m.weight.data, mean=1, std=0.02)
                if m.bias is not None:
                    init.constant_(m.bias.data, 0)
            elif classname.find('Linear') != -1:
                init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    init.normal_(m.bias.data)

    def forward(self, trainG, img_real, wf_real, color_real):
        """calculate loss inside forward to mitigate the imblance memory usage issue of multi-gpu training in pytorch"""

        if self.colorguide:
            color = self.enc_color(color_real.view(-1, 256, 3)).view(-1, 3, 256, 256)
            embedding = self.enc(torch.cat((wf_real, color),1))
        else:
            embedding = self.enc(wf_real)
        
        img_base = self.dec_base_img(embedding)
        wf_base = self.dec_base_wf(embedding)
        wf = self.dec_wf(wf_base)
        img_base = torch.cat((img_base, wf_base), 1)
        
        img = self.dec_img(img_base)
        if self.colorguide:
            color = self.dec_color(img.view(-1,256,256*3)).view(-1,256*3)
        del img_base, wf_base, embedding

        criterionL1 = nn.L1Loss().cuda()
        if trainG:
            lambdas = self.lambdas
            img_l1 = criterionL1(img, img_real.detach())
            wf_ssim = - msssim(wf, wf_real.detach(), normalize=True)   # normalize to stabilize training         
            wf_l1 = criterionL1(wf, wf_real.detach())
            if self.colorguide:
                color_l1 = criterionL1(color, color_real.detach())
                # overall loss except the perceptual loss
                lossG = img_l1 * lambdas[0] + wf_ssim * lambdas[2] + wf_l1 * lambdas[1] + color_l1 * lambdas[3]
            else:
                color_l1 = None
                lossG = img_l1 * lambdas[0] + wf_ssim * lambdas[2] + wf_l1 * lambdas[1]

        else:
            lossG, wf_ssim, img_l1, color_l1, wf_l1 = None, None, None, None, None
        
        return img, wf, lossG, wf_ssim, img_l1, color_l1, wf_l1



class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):

        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=4, norm_layer=SynchronizedBatchNorm2d):
        """Construct a PatchGAN discriminator"""
        super(NLayerDiscriminator, self).__init__()

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 16)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=False),
                norm_layer(ndf * nf_mult, affine=True),
                nn.LeakyReLU(0.2, True)
            ]

        kw = 3
        if n_layers == 5:
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n_layers, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=False),
                norm_layer(ndf * nf_mult, affine=True),
                nn.LeakyReLU(0.2, True)
            ]
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n_layers, 4)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=False),
                norm_layer(ndf * nf_mult, affine=True),
                nn.LeakyReLU(0.2, True)
            ]
        if n_layers == 4:
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n_layers, 4)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=False),
                norm_layer(ndf * nf_mult, affine=True),
                nn.LeakyReLU(0.2, True)
            ]

        kw = 4
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map, hard coded for now
        self.model = nn.Sequential(*sequence)

        # weight init
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv2d') != -1:
                init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    init.normal_(m.bias.data)
            elif classname.find('BatchNorm2d') != -1:
                init.normal_(m.weight.data, mean=1, std=0.02)
                if m.bias is not None:
                    init.constant_(m.bias.data, 0)

    def forward(self, trainG, trainReal, real_AB, fake_AB):
        """calculate loss inside forward to mitigate the imblance memory usage issue of multi-gpu training in pytorch"""

        # labels for training G/D nets
        real_label = 1
        fake_label = 0
        # by default, we use LSGAN loss
        criterionGAN = nn.MSELoss().cuda()
        # when training G
        if trainG:
            pred_fake =self.model(fake_AB).view(-1)
            label_real = torch.full(pred_fake.size(), real_label).cuda()
            errD = criterionGAN(pred_fake, label_real)
            del label_real, pred_fake
        # when training D, contains two steps
        else:
            if trainReal:
                pred_real = self.model(real_AB).view(-1)
                label_real = torch.full(pred_real.size(), real_label).cuda()
                errD = criterionGAN(pred_real, label_real)
                del label_real, pred_real
            else:
                pred_fake = self.model(fake_AB.detach()).view(-1)
                label_fake = torch.full(pred_fake.size(), fake_label).cuda()
                errD = criterionGAN(pred_fake, label_fake)
                del label_fake, pred_fake

        return errD
