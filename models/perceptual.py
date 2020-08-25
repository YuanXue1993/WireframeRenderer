""" code adapted from https://github.com/richzhang/PerceptualSimilarity """
import torch
import torch.nn as nn
from .vgg import vgg16


class PNet(nn.Module):
    '''Pre-trained network with all channels equally weighted by default'''
    def __init__(self, net_type='vgg', use_gpu=True):
        super(PNet, self).__init__()

        self.use_gpu = use_gpu

        # weights inside pretrained vgg shouldn't be updated, and also for saving memory
        self.vgg = vgg16(pretrained=True, requires_grad=False)
        self.vgg.eval()

        self.L = self.vgg.N_slices

        if(use_gpu):
            self.vgg.cuda()

    def forward(self, in0, in1, retPerLayer=False):
        # convert from -1~1 to 0~1
        in0 = in0 * 0.5 + 0.5
        in1 = in1 * 0.5 + 0.5

        # to ensure working properly on multi-gpus
        mean = torch.as_tensor([.229, .224, .225], dtype=in0.dtype, device=in0.device).view(1,3,1,1).expand_as(in0)
        std = torch.as_tensor([.485, .456, .406], dtype=in0.dtype, device=in0.device).view(1,3,1,1).expand_as(in0)
        in0_sc = in0.sub_(mean).div_(std)

        mean = torch.as_tensor([.229, .224, .225], dtype=in1.dtype, device=in1.device).view(1,3,1,1).expand_as(in1)
        std = torch.as_tensor([.485, .456, .406], dtype=in1.dtype, device=in1.device).view(1,3,1,1).expand_as(in1)
        in1_sc = in1.sub_(mean).div_(std)


        outs0 = self.vgg.forward(in0_sc)
        outs1 = self.vgg.forward(in1_sc)


        # calculate the L2 distance between two normalized inputs
        for (kk,out0) in enumerate(outs0):
            cur_score = (1.-self.cos_sim(outs0[kk],outs1[kk]))
            if(kk==0):
                val = 1.*cur_score
            else:
                val = val + cur_score

        return val.mean()

    def normalize_tensor(self, in_feat, eps=1e-10):
        norm_factor = torch.sqrt(torch.sum(in_feat**2,dim=1)).view(in_feat.size(0),1,in_feat.size(2),in_feat.size(3))
        return in_feat/(norm_factor.expand_as(in_feat)+eps)

    def cos_sim(self, in0, in1):
        in0_norm = self.normalize_tensor(in0)
        in1_norm = self.normalize_tensor(in1)
        N = in0.size(0)
        X = in0.size(2)
        Y = in0.size(3)

        return torch.mean(torch.mean(torch.sum(in0_norm*in1_norm,dim=1).view(N,1,X,Y),dim=2).view(N,1,1,Y),dim=3).view(N)