import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import models.archs.arch_util as arch_util

class ResNeXtblock(nn.Module):

    def __init__(self, nf, gc,cardinality, base_width, widen_factor):
        super(ResNeXtblock, self).__init__()
        width_ratio = gc / (widen_factor * 64.)
        D = cardinality * int(base_width * width_ratio)
        self.conv_reduce = nn.Conv2d(nf, D, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_conv = nn.Conv2d(D, D, kernel_size=3, stride=1, padding=1, bias=False, groups=cardinality)
        self.conv_expand = nn.Conv2d(D, gc, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu=nn.ReLU(inplace=True)
        arch_util.initialize_weights([self.conv_reduce,self.conv_conv,self.conv_expand],
                                     0.1)
    def forward(self, x):
        bottleneck = self.relu(self.conv_reduce(x))
        bottleneck = self.relu(self.conv_conv(bottleneck))
        bottleneck = self.conv_expand(bottleneck)
        return bottleneck*0.2 + x

class RRXD(nn.Module):

    def __init__(self, nf,gc,cardinality, base_width, widen_factor):
        super(RRXD, self).__init__()
        self.RRXD1 = ResNeXtblock(nf, gc,cardinality, base_width, widen_factor)
        self.RRXD2 = ResNeXtblock(gc,nf,  cardinality, base_width, widen_factor)

    def forward(self, x):
        out = self.RRXD1(x)
        out = self.RRXD2(out)
        return out * 0.2 + x

class RRXNet(nn.Module):
    def __init__(self,in_nc, out_nc,nf, nb,cardinality, base_width, widen_factor):
        super(RRXNet, self).__init__()
        # nf=ou
        RRDX_block_f = functools.partial(RRXD, nf=nf, gc=nf,cardinality=cardinality, base_width=base_width,widen_factor= widen_factor)

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDX_trunk = arch_util.make_layer(RRDX_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        #### upsampling
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)


    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDX_trunk(fea))
        fea=fea + trunk
        fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.HRconv(fea)))
        return out







