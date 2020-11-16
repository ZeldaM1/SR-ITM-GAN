import torch
# import models.archs.SRResNet_arch as SRResNet_arch
import models.archs.discriminator_vgg_arch as SRGAN_arch
import models.archs.RRDBNet_arch as RRDBNet_arch
# import models.archs.EDVR_arch as EDVR_arch


# Generator
def define_G(opt):
    opt_net = opt['network_G']
    which_model = opt_net['which_model_G']


    if which_model == 'RRDBNet':

        netG = RRDBNet_arch.RRXNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],nf=opt_net['nf'], nb=opt_net['nb'],
                                    cardinality=opt_net['cardinality'], base_width=opt_net['base_width'], widen_factor=opt_net['widen_factor'])

    else:
        raise NotImplementedError('Generator model [{:s}] not recognized'.format(which_model))

    return netG


# Discriminator
def define_D(opt):
    opt_net = opt['network_D']
    which_model = opt_net['which_model_D']

    if which_model == 'discriminator_vgg_128':
        netD = SRGAN_arch.Discriminator_VGG_128(in_nc=opt_net['in_nc'], nf=opt_net['nf'])
    elif which_model == 'vgg19_bn':
        netD = SRGAN_arch.VGG19_bn(pretrained=False)
    else:
        raise NotImplementedError('Discriminator model [{:s}] not recognized'.format(which_model))
    return netD


# Define network used for perceptual loss
def define_F(opt, use_bn=False):
    gpu_ids = opt['gpu_ids']
    device = torch.device('cuda' if gpu_ids else 'cpu')
    opt_net = opt['network_F']
    which_model = opt_net['which_model_F']
    if which_model == 'vgg19':
        # PyTorch pretrained VGG19-54, before ReLU.
        if use_bn:
            feature_layer = 49
        else:
            feature_layer = 34
        netF = SRGAN_arch.VGGFeatureExtractor(feature_layer=feature_layer, use_bn=use_bn,
                                              use_input_norm=True, device=device)
    elif which_model == 'vgg16':
        netF = SRGAN_arch.Vgg16(use_bn=use_bn)
    else:
        raise NotImplementedError('F net  [{:s}] not recognized'.format(which_model))

    # netF.eval()  # No need to train
    return netF


