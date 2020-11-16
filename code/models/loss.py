import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp,pi

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        return loss


# Define GAN loss: [vanilla | lsgan | wgan-gp]
class GANLoss(nn.Module):
    def __init__(self, gan_type, real_label_val=1.0, fake_label_val=0.0):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type.lower()
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

        if self.gan_type == 'gan' or self.gan_type == 'ragan':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif self.gan_type == 'wgan-gp':

            def wgan_loss(input, target):
                # target is boolean
                return -1 * input.mean() if target else input.mean()

            self.loss = wgan_loss
        else:
            raise NotImplementedError('GAN type [{:s}] is not found'.format(self.gan_type))

    def get_target_label(self, input, target_is_real):
        if self.gan_type == 'wgan-gp':
            return target_is_real
        if target_is_real:
            return torch.empty_like(input).fill_(self.real_label_val)
        else:
            return torch.empty_like(input).fill_(self.fake_label_val)

    def forward(self, input, target_is_real):
        target_label = self.get_target_label(input, target_is_real)
        loss = self.loss(input, target_label)
        return loss


class GradientPenaltyLoss(nn.Module):
    def __init__(self, device=torch.device('cpu')):
        super(GradientPenaltyLoss, self).__init__()
        self.register_buffer('grad_outputs', torch.Tensor())
        self.grad_outputs = self.grad_outputs.to(device)

    def get_grad_outputs(self, input):
        if self.grad_outputs.size() != input.size():
            self.grad_outputs.resize_(input.size()).fill_(1.0)
        return self.grad_outputs

    def forward(self, interp, interp_crit):
        grad_outputs = self.get_grad_outputs(interp_crit)
        grad_interp = torch.autograd.grad(outputs=interp_crit, inputs=interp,
                                          grad_outputs=grad_outputs, create_graph=True,
                                          retain_graph=True, only_inputs=True)[0]
        grad_interp = grad_interp.view(grad_interp.size(0), -1)
        grad_interp_norm = grad_interp.norm(2, dim=1)

        loss = ((grad_interp_norm - 1)**2).mean()
        return loss

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255#65535
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret


def msssim(img1, img2, window_size=11, size_average=True, val_range=None, normalize=False):
    device = img1.device
    #weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(device)
    weights = torch.FloatTensor([0.0717,0.4529,0.4759]).to(device)

    levels = weights.size()[0]
    mssim = []
    mcs = []
    for _ in range(levels):
        sim, cs = ssim(img1, img2, window_size=window_size, size_average=size_average, full=True, val_range=val_range)
        mssim.append(sim)
        mcs.append(cs)

        img1 = F.avg_pool2d(img1, (2, 2))
        img2 = F.avg_pool2d(img2, (2, 2))

    mssim = torch.stack(mssim)
    mcs = torch.stack(mcs)

    # Normalize (to avoid NaNs during training unstable models, not compliant with original definition)
    if normalize:
        mssim = (mssim + 1) / 2
        mcs = (mcs + 1) / 2

    pow1 = mcs ** weights
    pow2 = mssim ** weights
    # From Matlab implementation https://ece.uwaterloo.ca/~z70wang/research/iwssim/
    output = torch.prod(pow1[:-1] * pow2[-1])
    return output


# Classes to re-use window
class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range

        # Assume 1 channel for SSIM
        self.channel = 1
        self.window = create_window(window_size)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel

        return ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)

class MSSSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, channel=3):
        super(MSSSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel

    def forward(self, img1, img2):
        # TODO: store window between calls if possible
        msssim_loss=msssim(img1, img2, window_size=self.window_size, size_average=self.size_average)
        return (1-msssim_loss)#最大化ms-ssim

def gram(x):
    (bs, ch, h, w) = x.size()
    f = x.view(bs, ch, w * h)
    f_T = f.transpose(1, 2)
    G = f.bmm(f_T) / (ch * h * w)
    return G


class PerceptualLoss(nn.Module):
    def __init__(self):#,x,y_hat,shape
        super(PerceptualLoss, self).__init__()
        # self.content_layer=2
        self.mse = nn.DataParallel(nn.MSELoss())
        self.mse_sum = nn.DataParallel(nn.MSELoss(reduction='sum'))
        self.l1_loss = nn.DataParallel(nn.L1Loss())


    def forward(self, x, y_hat,y_content_features ,y_hat_features,style_features,
                STYLE_WEIGHT,CONTENT_WEIGHT,TV_WEIGHT):

       # style_image=y_hat


        with torch.no_grad():
        #    self.style_features = self.netF(style_image)
            self.style_gram = [gram(fmap) for fmap in style_features]
        pass
        b, c, h, w = x.shape
        #y_content_features = self.netF(x)
        #y_hat_features = self.netF(y_hat)

        recon = y_content_features[2]#self.content_layer]  # 2
        recon_hat = y_hat_features[2]#self.content_layer]  # 2
        L_content = self.mse(recon_hat, recon)

        y_hat_gram = [gram(fmap) for fmap in y_hat_features]
        L_style = 0
        for j in range(len(y_content_features)):
            _, c_l, h_l, w_l = y_hat_features[j].shape
            L_style += self.mse_sum(y_hat_gram[j], self.style_gram[j]) / float(c_l * h_l * w_l)

        # L_pixel = self.mse(y_hat, x)
        L_pixel = self.l1_loss(y_hat, x)

        # calculate total variation regularization (anisotropic version)
        # https://www.wikiwand.com/en/Total_variation_denoising
        diff_i = torch.sum(torch.abs(y_hat[:, :, :, 1:] - y_hat[:, :, :, :-1]))
        diff_j = torch.sum(torch.abs(y_hat[:, :, 1:, :] - y_hat[:, :, :-1, :]))
        L_tv = (diff_i + diff_j) / float(c * h * w)

        perceptual_loss = STYLE_WEIGHT * L_style + CONTENT_WEIGHT * (L_content + L_pixel) + L_tv * TV_WEIGHT
        l_g_fea = perceptual_loss.mean()
        # return L_content, L_style, L_pixel, L_tv
        return l_g_fea

class Color_loss(nn.Module):
    def __init__(self):#,x,y_hat,shape
        super(Color_loss, self).__init__()

    def forward(self,fake_H, label):
        #[n,h,w,c]  tf
        #b, c, h, w = x.shape  pytorch  0 1 2 3 -> 0 2 3 1
        # fake_H=fake_H.transpose(0,2,3,1)
        # label=label.transpose(0,2,3,1)

        # b, c, h, w = fake_H.shape
        # print(fake_H.ndim)
        vec1 = fake_H.reshape(-1,3)
        vec2 = label.reshape(-1, 3)
        clip_value = 0.999999
        norm_vec1 = F.normalize(vec1, p=2,dim=1)
        norm_vec2 = F.normalize(vec2, p=2,dim=1)
        dot = torch.mean(norm_vec1 * norm_vec2,dim= 1)#tf.reduce_sum(norm_vec1*norm_vec2, 1)
        dot = torch.clamp(dot, min=-clip_value, max=clip_value)
        angle = torch.acos(dot) * (180 / pi)

        return torch.mean(angle)


# def smoothness_loss(image):
#     clip_low, clip_high = 0.000001, 0.999999
#     image = tf.clip_by_value(image, clip_low, clip_high)
#     image_h, image_w = tf.shape(image)[1], tf.shape(image)[2]
#     tv_x = tf.reduce_mean((image[:, 1:, :, :] - image[:, :image_h - 1, :, :]) ** 2)
#     tv_y = tf.reduce_mean((image[:, :, 1:, :] - image[:, :, :image_w - 1, :]) ** 2)
#     total_loss = (tv_x + tv_y) / 2
#     '''
#     log_image = tf.log(image)
#     log_tv_x = tf.reduce_mean((log_image[:, 1:, :, :]-
#                               log_image[:, :image_h-1, :, :])**1.2)
#     log_tv_y = tf.reduce_mean((log_image[:, :, 1:, :]-
#                                log_image[:, :, :image_w-1, :])**1.2)
#     total_loss = tv_x / (log_tv_x + 1e-4) + tv_y / (log_tv_y + 1e-4)
#     '''
#     return total_loss
#
#
# def reconstruct_loss(image, label):
#     l2_loss = tf.reduce_mean(tf.square(label - image))
#     return l2_loss
