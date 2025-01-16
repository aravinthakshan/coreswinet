import torch
import torchvision
import cv2
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import torchvision.transforms as T
from torch.autograd import Variable
import torch.autograd as autograd
import math 
import torch
import torch.nn as nn
from segmentation_models_pytorch.utils import base
import torch.autograd as autograd
from torch.autograd import Variable
import numpy as np
import functools
import torch.nn as nn

class TextureLoss(nn.Module):
    def __init__(self):
        super(TextureLoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def gram_matrix(self, features):
        """
        Compute the Gram matrix for a set of features.
        Args:
            features (torch.Tensor): Feature maps of shape (B, C, H, W).
        Returns:
            torch.Tensor: Gram matrix of shape (B, C, C).
        """
        B, C, H, W = features.size()
        # Reshape features to (B, C, H*W)
        features = features.view(B, C, -1)
        # Compute the Gram matrix
        gram = torch.bmm(features, features.transpose(1, 2))
        # Normalize the Gram matrix
        gram /= C * H * W
        return gram

    def forward(self, generated_features, target_features):
        """
        Compute the texture loss between generated and target features.
        Args:
            generated_features (torch.Tensor): Features from the generated image.
            target_features (torch.Tensor): Features from the target image.
        Returns:
            torch.Tensor: Texture loss value.
        """
        gram_generated = self.gram_matrix(generated_features)
        gram_target = self.gram_matrix(target_features)
        loss = self.mse_loss(gram_generated, gram_target)
        return loss

def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')

def mae_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')

class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.blocks = torch.nn.ModuleList(blocks).to(device)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device))

    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss


class ShallowLoss(nn.Module):
    def __init__(self, edgeType, lossType):
        super(ShallowLoss, self).__init__()
        assert edgeType in ('canny', 'dog')
        self.edgeType = edgeType
        assert lossType in ('mse', 'mae')
        self.lossType = lossType
        
    def forward(self, shallow_op, gt):
        if self.edgeType == 'canny':
            gt_np = gt.squeeze().cpu().numpy()
            
            median = np.median(gt_np)
            lower_threshold = int(max(0, median * 0.66))
            upper_threshold = int(min(255, median * 1.33)) 
            edges = cv2.Canny(gt_np.astype(np.uint8), threshold1=lower_threshold, threshold2=upper_threshold)
            edges_normalized = edges / 255.0

        elif self.edgeType == 'dog':
            gt = rgb_to_grayscale(gt).unsqueeze(1)
            gk1 = gaussian_kernel(1).repeat(gt.shape[0],1,1,1)
            gk2 = gaussian_kernel(2).repeat(gt.shape[0],1,1,1)
            
            gt = gt.permute(1, 0, 2, 3)
            g1 = torch.nn.functional.conv2d(gt, gk1.to(gt.device), groups=gt.shape[1])
            g2 = torch.nn.functional.conv2d(gt, gk2.to(gt.device), groups=gt.shape[1])

            dog = g1-g2
            
            dog_np = dog.detach().cpu().numpy()
            dog_np = (dog_np).astype(np.uint8)


            _, edges = cv2.threshold(dog_np, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            edges_normalized = edges / 255.0

        edges_tensor = torch.tensor(edges_normalized, dtype=torch.float32, device=shallow_op.device)
        edges_tensor = edges_tensor
        
        edges_normalized = edges_tensor.repeat(1, 3, 1, 1)
        edges_normalized = F.interpolate(edges_normalized, size=shallow_op.shape[2:], mode='bilinear', align_corners=False)

        # Calculate loss (MSE or MAE)
        if self.lossType == 'mse':
            return mse_loss(shallow_op, edges_normalized)
        elif self.lossType == 'mae':
            return mae_loss(shallow_op, edges_normalized)
        
def gaussian_kernel(sigma):
    """Generates a Gaussian kernel as a PyTorch tensor."""
    if sigma == 1:
        return (torch.tensor([
        [0.0039, 0.0156, 0.0235, 0.0156, 0.0039],
        [0.0156, 0.0625, 0.0938, 0.0625, 0.0156],
        [0.0235, 0.0938, 0.1406, 0.0938, 0.0235],
        [0.0156, 0.0625, 0.0938, 0.0625, 0.0156],
        [0.0039, 0.0156, 0.0235, 0.0156, 0.0039]
    ]))
    if sigma == 2:
        return (torch.tensor([
        [0.0002, 0.0004, 0.0006, 0.0004, 0.0002],
        [0.0004, 0.0009, 0.0014, 0.0009, 0.0004],
        [0.0006, 0.0014, 0.0023, 0.0014, 0.0006],
        [0.0004, 0.0009, 0.0014, 0.0009, 0.0004],
        [0.0002, 0.0004, 0.0006, 0.0004, 0.0002]
    ]))

def rgb_to_grayscale(image):
    return 0.2989 * image[:, 0, :, :] + 0.5870 * image[:, 1, :, :] + 0.1140 * image[:, 2, :, :]

class GANLoss(nn.Module):
    """Define GAN loss.

    Args:
        gan_type (str): Support 'vanilla', 'lsgan', 'wgan', 'hinge'.
        real_label_val (float): The value for real label. Default: 1.0.
        fake_label_val (float): The value for fake label. Default: 0.0.
        loss_weight (float): Loss weight. Default: 1.0.
            Note that loss_weight is only for generators; and it is always 1.0
            for discriminators.
    """

    def __init__(self,
                 gan_type,
                 real_label_val=1.0,
                 fake_label_val=0.0,
                 loss_weight=1.0,
                 device='cuda'):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type
        self.loss_weight = loss_weight
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val
        self.device = device

        if self.gan_type == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == 'standard':
            self.loss = None
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif self.gan_type == 'wgan':
            self.loss = self._wgan_loss
        elif self.gan_type == 'wgan_softplus':
            self.loss = self._wgan_softplus_loss
        elif self.gan_type == 'hinge':
            self.loss = nn.ReLU()
        else:
            raise NotImplementedError(
                f'GAN type {self.gan_type} is not implemented.')

    def _wgan_loss(self, input, target):
        """wgan loss.

        Args:
            input (Tensor): Input tensor.
            target (bool): Target label.

        Returns:
            Tensor: wgan loss.
        """
        return -input.mean() if target else input.mean()

    def _wgan_softplus_loss(self, input, target):
        """wgan loss with soft plus. softplus is a smooth approximation to the
        ReLU function.

        In StyleGAN2, it is called:
            Logistic loss for discriminator;
            Non-saturating loss for generator.

        Args:
            input (Tensor): Input tensor.
            target (bool): Target label.

        Returns:
            Tensor: wgan loss.
        """
        return F.softplus(-input).mean() if target else F.softplus(
            input).mean()

    def get_target_label(self, input, target_is_real):
        """Get target label.

        Args:
            input (Tensor): Input tensor.
            target_is_real (bool): Whether the target is real or fake.

        Returns:
            (bool | Tensor): Target tensor. Return bool for wgan, otherwise,
                return Tensor.
        """

        if self.gan_type in ['wgan', 'wgan_softplus']:
            return target_is_real
        target_val = (
            self.real_label_val if target_is_real else self.fake_label_val)
        return input.new_ones(input.size()) * target_val

    def forward(self, input, target_is_real, is_disc=False):
        """
        Args:
            input (Tensor): The input for the loss module, i.e., the network
                prediction.
            target_is_real (bool): Whether the targe is real or fake.
            is_disc (bool): Whether the loss for discriminators or not.
                Default: False.

        Returns:
            Tensor: GAN loss value.
        """
        target_label = self.get_target_label(input, target_is_real)
        if self.gan_type == 'standard':
            if is_disc:
                if target_is_real:
                    loss = -torch.mean(input)
                else:
                    loss = torch.mean(input)
            else:
                loss = -torch.mean(input)
        elif self.gan_type == 'hinge':
            if is_disc:  # for discriminators in hinge-gan
                input = -input if target_is_real else input
                loss = self.loss(1 + input).mean()
            else:  # for generators in hinge-gan
                loss = -input.mean()
        else:  # other gan types
            loss = self.loss(input, target_label)

        # loss_weight is always 1.0 for discriminators
        return loss if is_disc else loss * self.loss_weight

def r1_penalty(real_pred, real_img):
    """R1 regularization for discriminator. The core idea is to
        penalize the gradient on real data alone: when the
        generator distribution produces the true data distribution
        and the discriminator is equal to 0 on the data manifold, the
        gradient penalty ensures that the discriminator cannot create
        a non-zero gradient orthogonal to the data manifold without
        suffering a loss in the GAN game.

        Ref:
        Eq. 9 in Which training methods for GANs do actually converge.
        """
    grad_real = autograd.grad(
        outputs=real_pred.sum(), inputs=real_img, create_graph=True)[0]
    grad_penalty = grad_real.pow(2).view(grad_real.shape[0], -1).sum(1).mean()
    return grad_penalty


def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):
    noise = torch.randn_like(fake_img) / math.sqrt(
        fake_img.shape[2] * fake_img.shape[3])
    grad = autograd.grad(
        outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True)[0]
    path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

    path_mean = mean_path_length + decay * (
        path_lengths.mean() - mean_path_length)

    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_lengths.detach().mean(), path_mean.detach()


def compute_gradient_penalty(D, real_samples, fake_samples, device):

    # Random weight term for interpolation between real and fake samples
    alpha = torch.FloatTensor(np.random.random((real_samples.size(0), 1, 1, 1))).to(device)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    interpolates = interpolates.to(device)
    d_interpolates = D(interpolates)
    d_interpolates = d_interpolates.to(device)
    fake = Variable(torch.FloatTensor(real_samples.shape[0], 1, 1, 1).fill_(1.0), requires_grad=False)
    fake = fake.to(device)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


class PSNRLoss(nn.Module):
    def __init__(self, max_val=1.0):
        """
        Initialize PSNR loss module
        Args:
            max_val (float): Maximum value of the input (default=1.0 for normalized images)
        """
        super(PSNRLoss, self).__init__()
        self.max_val = max_val

    def forward(self, pred, target):
        """
        Calculate PSNR loss scaled to approximately match SSIM range
        Args:
            pred (torch.Tensor): Predicted images
            target (torch.Tensor): Target images
        Returns:
            torch.Tensor: Scaled PSNR loss
        """
        # Calculate MSE
        mse = torch.mean((pred - target) ** 2, dim=[1, 2, 3])
        
        # Calculate PSNR
        psnr = 20 * torch.log10(self.max_val) - 10 * torch.log10(mse)
        
        # Scale PSNR to approximate SSIM range (0-1)
        # Typical PSNR values are around 20-40 dB, so dividing by 50 maps this to 0.4-0.8
        scaled_psnr = psnr / 50.0
        
        # Return negative for loss minimization
        return -scaled_psnr.mean()
    
class CharbonnierLoss(nn.Module):
    """Charbonnier loss (a differentiable variant of L1Loss).
    
    Args:
        loss_weight (float): Loss weight for the Charbonnier loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
        eps (float): A value used to control the curvature near zero.
            Default: 1e-12.
    """

    def __init__(self, loss_weight=1.0, reduction='mean', eps=1e-12):
        super(CharbonnierLoss, self).__init__()
        
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: "none", "mean", "sum".')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.eps = eps

    def forward(self, pred, target, weight=None):
        """
        Args:
            pred (Tensor): Predicted tensor of shape (N, C, H, W).
            target (Tensor): Ground truth tensor of shape (N, C, H, W).
            weight (Tensor, optional): Element-wise weights. Default: None.
        """
        # Compute the Charbonnier loss
        diff = pred - target
        loss = torch.sqrt(diff ** 2 + self.eps ** 2)

        if weight is not None:
            loss = loss * weight

        # Apply the specified reduction mode
        if self.reduction == 'mean':
            return self.loss_weight * loss.mean()
        elif self.reduction == 'sum':
            return self.loss_weight * loss.sum()
        else:  # 'none'
            return self.loss_weight * loss


def reduce_loss(loss, reduction):
    """Reduce loss as specified.

    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are 'none', 'mean' and 'sum'.

    Returns:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    else:
        return loss.sum()


def weight_reduce_loss(loss, weight=None, reduction='mean'):
    """Apply element-wise weight and reduce loss.

    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights. Default: None.
        reduction (str): Same as built-in losses of PyTorch. Options are
            'none', 'mean' and 'sum'. Default: 'mean'.

    Returns:
        Tensor: Loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        assert weight.dim() == loss.dim()
        assert weight.size(1) == 1 or weight.size(1) == loss.size(1)
        loss = loss * weight

    # if weight is not specified or reduction is sum, just reduce the loss
    if weight is None or reduction == 'sum':
        loss = reduce_loss(loss, reduction)
    # if reduction is mean, then compute mean over weight region
    elif reduction == 'mean':
        if weight.size(1) > 1:
            weight = weight.sum()
        else:
            weight = weight.sum() * loss.size(1)
        loss = loss.sum() / weight

    return loss


def weighted_loss(loss_func):
    """Create a weighted version of a given loss function.

    To use this decorator, the loss function must have the signature like
    `loss_func(pred, target, **kwargs)`. The function only needs to compute
    element-wise loss without any reduction. This decorator will add weight
    and reduction arguments to the function. The decorated function will have
    the signature like `loss_func(pred, target, weight=None, reduction='mean',
    **kwargs)`."""

    @functools.wraps(loss_func)
    def wrapper(pred, target, weight=None, reduction='mean', **kwargs):
        # get element-wise loss
        loss = loss_func(pred, target, **kwargs)
        loss = weight_reduce_loss(loss, weight, reduction)
        return loss

    return wrapper

def device_as(t1, t2):
    """
    Moves t1 to the device of t2
    """
    return t1.to(t2.device)
    
class ContrastiveLoss(nn.Module):
    """
    Vanilla Contrastive loss, also called InfoNceLoss as in SimCLR paper
    """
    def __init__(self, batch_size, temperature=0.5):
        super().__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.mask = (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float()

    def calc_similarity_batch(self, a, b):
        representations = torch.cat([a, b], dim=0)
        return F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)

    def forward(self, proj_1, proj_2):
        """
        proj_1 and proj_2 are batched embeddings [batch, embedding_dim]
        where corresponding indices are pairs
        z_i, z_j in the SimCLR paper
        """
        batch_size = proj_1.shape[0]
        z_i = F.normalize(proj_1, p=2, dim=1)
        z_j = F.normalize(proj_2, p=2, dim=1)

        similarity_matrix = self.calc_similarity_batch(z_i, z_j)

        sim_ij = torch.diag(similarity_matrix, batch_size)
        sim_ji = torch.diag(similarity_matrix, -batch_size)

        positives = torch.cat([sim_ij, sim_ji], dim=0)

        nominator = torch.exp(positives / self.temperature)

        denominator = device_as(self.mask, similarity_matrix) * torch.exp(similarity_matrix / self.temperature)

        all_losses = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(all_losses) / (2 * self.batch_size)
        return loss

@weighted_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')

class MSELoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MSELoss, self).__init__()

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * mse_loss(
            pred, target, weight, reduction=self.reduction)

class GANLoss(nn.Module):
    """Define GAN loss.

    Args:
        gan_type (str): Support 'vanilla', 'lsgan', 'wgan', 'hinge'.
        real_label_val (float): The value for real label. Default: 1.0.
        fake_label_val (float): The value for fake label. Default: 0.0.
        loss_weight (float): Loss weight. Default: 1.0.
            Note that loss_weight is only for generators; and it is always 1.0
            for discriminators.
    """

    def __init__(self,
                 gan_type='standard',
                 real_label_val=1.0,
                 fake_label_val=0.0,
                 loss_weight=1.0,
                 device='cuda'):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type
        self.loss_weight = loss_weight
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val
        self.device = device

        if self.gan_type == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == 'standard':
            self.loss = None
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif self.gan_type == 'wgan':
            self.loss = self._wgan_loss
        elif self.gan_type == 'wgan_softplus':
            self.loss = self._wgan_softplus_loss
        elif self.gan_type == 'hinge':
            self.loss = nn.ReLU()
        else:
            raise NotImplementedError(
                f'GAN type {self.gan_type} is not implemented.')

    def _wgan_loss(self, input, target):
        """wgan loss.

        Args:
            input (Tensor): Input tensor.
            target (bool): Target label.

        Returns:
            Tensor: wgan loss.
        """
        return -input.mean() if target else input.mean()

    def _wgan_softplus_loss(self, input, target):
        """wgan loss with soft plus. softplus is a smooth approximation to the
        ReLU function.

        In StyleGAN2, it is called:
            Logistic loss for discriminator;
            Non-saturating loss for generator.

        Args:
            input (Tensor): Input tensor.
            target (bool): Target label.

        Returns:
            Tensor: wgan loss.
        """
        return F.softplus(-input).mean() if target else F.softplus(
            input).mean()

    def get_target_label(self, input, target_is_real):
        """Get target label.

        Args:
            input (Tensor): Input tensor.
            target_is_real (bool): Whether the target is real or fake.

        Returns:
            (bool | Tensor): Target tensor. Return bool for wgan, otherwise,
                return Tensor.
        """

        if self.gan_type in ['wgan', 'wgan_softplus']:
            return target_is_real
        target_val = (
            self.real_label_val if target_is_real else self.fake_label_val)
        return input.new_ones(input.size()) * target_val

    def forward(self, input, target_is_real, is_disc=False):
        """
        Args:
            input (Tensor): The input for the loss module, i.e., the network
                prediction.
            target_is_real (bool): Whether the targe is real or fake.
            is_disc (bool): Whether the loss for discriminators or not.
                Default: False.

        Returns:
            Tensor: GAN loss value.
        """
        target_label = self.get_target_label(input, target_is_real)
        if self.gan_type == 'standard':
            if is_disc:
                if target_is_real:
                    loss = -torch.mean(input)
                else:
                    loss = torch.mean(input)
            else:
                loss = -torch.mean(input)
        else:  # other gan types
            loss = self.loss(input, target_label)

        # loss_weight is always 1.0 for discriminators
        return loss if is_disc else loss * self.loss_weight

def compute_gradient_penalty(D, real_samples, fake_samples, device):

    # Random weight term for interpolation between real and fake samples
    alpha = torch.FloatTensor(np.random.random((real_samples.size(0), 1, 1, 1))).to(device)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    interpolates = interpolates.to(device)
    d_interpolates = D(interpolates)
    d_interpolates = d_interpolates.to(device)
    fake = Variable(torch.FloatTensor(real_samples.shape[0], 1, 1, 1).fill_(1.0), requires_grad=False)
    fake = fake.to(device)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

class custom_loss(base.Loss):
    def __init__(self, batch_size, beta=0.5, loss_weight=0.5, gan_type='standard'):
        super().__init__()

        self.gan_type = gan_type
        self.contrast = ContrastiveLoss(batch_size)
        self.GANLoss = GANLoss(self.gan_type)
        self.mse = nn.MSELoss()
        
        self.loss_weight = loss_weight
        self.beta = beta

    def forward(self, y_pr, y_gt, ft1=None, ft2=None):
        """
        Args:
            y_pr: predicted image
            y_gt: ground truth image
            ft1: feature map of predicted image
            ft2: feature map of ground truth image
        """
        c = self.contrast(ft1, ft2)
        g = self.GANLoss(y_pr, y_gt, is_disc=False)
        m = self.mse(y_pr, y_gt)

        return (self.beta)*c + (self.loss_weight)*m + (1-self.loss_weight)*g
    
class custom_loss_val(base.Loss):
    """
    Custom loss function for validation which DOESNT use contrastive loss.
    This is because contrastive loss is not differentiable.
    """
    def __init__(self, loss_weight=0.5, gan_type='standard'):
        super().__init__()

        self.gan_type = gan_type
        self.GANLoss = GANLoss(self.gan_type)
        self.mse = nn.MSELoss()
        
        self.loss_weight = loss_weight

    def forward(self, y_pr, y_gt):
        """
        Args:
            y_pr: predicted image
            y_gt: ground truth image
        """
        g = self.GANLoss(y_pr, y_gt, is_disc=False)
        m = self.mse(y_pr, y_gt)

        return (self.loss_weight)*m + (1-self.loss_weight)*g
        