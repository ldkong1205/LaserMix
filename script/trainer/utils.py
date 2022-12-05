from typing import Optional
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _WeightedLoss
from torchvision.transforms import functional as FF
from torch.autograd import Variable

try:
    from itertools import ifilterfalse
except ImportError:
    from itertools import filterfalse as ifilterfalse

from data.semantickitti.laserscan import LaserScan


inv_label_dict = {0: 0, 1: 10, 2: 11, 3: 15, 4: 18, 5: 20, 6: 30, 7: 31, 8: 32, 9: 40, 10: 44, 11: 48, 12: 49, 13: 50, 14: 51, 15: 70, 16: 71, 17: 72, 18: 80, 19: 81}



def postprocess_knn(A, pred, depth, knn, device):
    t_1 = torch.squeeze(depth*80.0).detach().to(device)
    t_2 = torch.squeeze(FF.to_tensor(np.reshape(A.unproj_range, (1, -1)))).detach().to(device)
    t_3 = torch.squeeze(pred).detach().to(device)
    t_4 = torch.squeeze(FF.to_tensor(np.reshape(A.proj_x, (1, -1)))).to(dtype=torch.long).detach().to(device)
    t_5 = torch.squeeze(FF.to_tensor(np.reshape(A.proj_y, (1, -1)))).to(dtype=torch.long).detach().to(device)

    unproj_argmax = knn(t_1, t_2, t_3, t_4, t_5)
    unproj_argmax = unproj_argmax.detach().cpu().numpy()

    label = []
    for i in unproj_argmax:
        upper_half = 0
        lower_half = inv_label_dict[i.item()]
        label_each = (upper_half << 16) + lower_half
        label.append(label_each)
    
    return label


def postprocess_fid(A, pred, depth, device):
    t_1 = torch.squeeze(depth*80.0).detach().to(device)
    t_3 = torch.squeeze(pred).detach().to(device)

    proj_unfold_range, proj_unfold_pre = NN_filter(t_1, t_3)
    proj_unfold_range = proj_unfold_range.cpu().numpy()
    proj_unfold_pre = proj_unfold_pre.cpu().numpy()

    label = []
    for jj in range(len(A.proj_x)):
        y_range, x_range = A.proj_y[jj], A.proj_x[jj]
        upper_half = 0
        if A.unproj_range[jj] == A.proj_range[y_range, x_range]:
            lower_half = inv_label_dict[pred[y_range, x_range]]
        else:
            potential_label = proj_unfold_pre[0, :, y_range, x_range]
            potential_range = proj_unfold_range[0, :, y_range, x_range]
            min_arg = np.argmin(abs(potential_range-A.unproj_range[jj]))
            lower_half = inv_label_dict[potential_label[min_arg]]
        label_each = (upper_half << 16) + lower_half
        label.append(label_each)

    return label


def NN_filter(depth, pred, k_size=5):
    pred = pred.double()
    H, W = np.shape(depth)
    
    proj_range_expand = torch.unsqueeze(depth, axis=0)
    proj_range_expand = torch.unsqueeze(proj_range_expand, axis=0)
    
    semantic_pred_expand = torch.unsqueeze(pred, axis=0)
    semantic_pred_expand = torch.unsqueeze(semantic_pred_expand, axis=0)
    
    pad = int((k_size - 1) / 2)

    proj_unfold_range = F.unfold(proj_range_expand,kernel_size=(k_size, k_size), padding=(pad, pad))
    proj_unfold_range = proj_unfold_range.reshape(-1, k_size*k_size, H, W)
        
    proj_unfold_pre = F.unfold(semantic_pred_expand,kernel_size=(k_size, k_size), padding=(pad, pad))
    proj_unfold_pre = proj_unfold_pre.reshape(-1, k_size*k_size, H, W)
    
    return proj_unfold_range, proj_unfold_pre


def get_gaussian_kernel(kernel_size=3, sigma=2, channels=1):
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()
    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.

    gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                      torch.exp(-torch.sum((xy_grid - mean) ** 2., dim=-1) / (2 * variance))
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
    gaussian_kernel = gaussian_kernel.view(kernel_size, kernel_size)

    return gaussian_kernel


class KNN(torch.nn.Module):
    def __init__(self, params={'knn': 5, 'search': 5, 'sigma': 1.0, 'cutoff': 1.0}, nclasses=20):
        super().__init__()
        self.knn = params["knn"]
        self.search = params["search"]
        self.sigma = params["sigma"]
        self.cutoff = params["cutoff"]
        self.nclasses = nclasses

    def forward(self, proj_range, unproj_range, proj_argmax, px, py):
        if proj_range.is_cuda: device = torch.device("cuda")
        else: device = torch.device("cpu")

        H, W = proj_range.shape
        P = unproj_range.shape

        if (self.search % 2 == 0): raise ValueError("Nearest neighbor kernel must be odd number")

        pad = int((self.search - 1) / 2)
        proj_unfold_k_rang = F.unfold(proj_range[None, None, ...], kernel_size=(self.search, self.search), padding=(pad, pad))

        idx_list = py * W + px

        unproj_unfold_k_rang = proj_unfold_k_rang[:, :, idx_list]
        unproj_unfold_k_rang[unproj_unfold_k_rang < 0] = float("inf")
        center = int(((self.search * self.search) - 1) / 2)
        unproj_unfold_k_rang[:, center, :] = unproj_range

        k2_distances = torch.abs(unproj_unfold_k_rang - unproj_range)
        inv_gauss_k = (1 - get_gaussian_kernel(self.search, self.sigma, 1)).view(1, -1, 1)
        inv_gauss_k = inv_gauss_k.to(device).type(proj_range.type())
        k2_distances = k2_distances * inv_gauss_k

        _, knn_idx = k2_distances.topk(self.knn, dim=1, largest=False, sorted=False)

        proj_unfold_1_argmax = F.unfold(proj_argmax[None, None, ...].float(), kernel_size=(self.search, self.search), padding=(pad, pad)).long()
        unproj_unfold_1_argmax = proj_unfold_1_argmax[:, :, idx_list]
        knn_argmax = torch.gather(input=unproj_unfold_1_argmax, dim=1, index=knn_idx)

        if self.cutoff > 0:
          knn_distances = torch.gather(input=k2_distances, dim=1, index=knn_idx)
          knn_invalid_idx = knn_distances > self.cutoff
          knn_argmax[knn_invalid_idx] = self.nclasses

        knn_argmax_onehot = torch.zeros((1, self.nclasses + 1, P[0]), device=device).type(proj_range.type())
        ones = torch.ones_like(knn_argmax).type(proj_range.type())
        knn_argmax_onehot = knn_argmax_onehot.scatter_add_(1, knn_argmax, ones)
        knn_argmax_out = knn_argmax_onehot[:, 1:-1].argmax(dim=1) + 1
        knn_argmax_out = knn_argmax_out.view(P)

        return knn_argmax_out


class ClassWeightSemikitti:
	@staticmethod
	def get_weight():
		return [0.0,
                1.0/(0.040818519255974316+0.001789309418528068+0.001),
                1.0/(0.00016609538710764618+0.001),
                1.0/(0.00039838616015114444+0.001),
                1.0/(0.0020633612104619787+0.00010157861367183268+0.001),
                1.0/(2.7879693665067774e-05+0.0016218197275284021+0.00011351574470342043+4.3840131989471124e-05+0.001),
                1.0/(0.00017698551338515307+0.00016059776092534436+0.001),
                1.0/(1.1065903904919655e-08+0.00012709999297008662+0.001),
                1.0/(5.532951952459828e-09+3.745553104802113e-05+0.001),
                1.0/(0.1987493871255525+4.7084144280367186e-05+0.001),
                1.0/(0.014717169549888214+0.001),
                1.0/(0.14392298360372+0.001),
                1.0/(0.0039048553037472045+0.001),
                1.0/(0.1326861944777486+0.001),
                1.0/(0.0723592229456223+0.001),
                1.0/(0.26681502148037506+0.001),
                1.0/(0.006035012012626033+0.001),
                1.0/(0.07814222006271769+0.001),
                1.0/(0.002855498193863172+0.001),
                1.0/(0.0006155958086189918+0.001)
            ]
	@staticmethod
	def get_bin_weight(bin_num):
		weight_list=[]
		for i in range(bin_num+1):
			weight_list.append(abs(i/float(bin_num)-0.5)*2+0.2)
		return weight_list


def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


def get_semantic_segmentation(sem):
	# map semantic output to labels
	if sem.size(0) != 1:
		raise ValueError('Only supports inference for batch size = 1')
	sem = sem.squeeze(0)
	predict_pre = torch.argmax(sem, dim=0, keepdim=True)
	'''
	sem_prob=F.softmax(sem,dim=0)
	change_mask_motorcyclist=torch.logical_and(predict_pre==7,sem_prob[8:9,:,:]>0.1)
	predict_pre[change_mask_motorcyclist]=8
	'''
	return predict_pre


def isnan(x):
    return x != x


def mean(l, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = ifilterfalse(isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_softmax(probas, labels, classes='present', per_image=False, ignore=None):
    """
    Multi-class Lovasz-Softmax loss
      probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1).
              Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
      labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
      per_image: compute the loss per image instead of per batch
      ignore: void class labels
    """
    if per_image:
        loss = mean(lovasz_softmax_flat(*flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore), classes=classes)
                    for prob, lab in zip(probas, labels))
    else:
        loss = lovasz_softmax_flat(*flatten_probas(probas, labels, ignore), classes=classes)
    return loss


def lovasz_softmax_flat(probas, labels, classes='present'):
    """
    Multi-class Lovasz-Softmax loss
      probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [P] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
    """
    if probas.numel() == 0:
        # only void pixels, the gradients should be 0
        return probas * 0.
    C = probas.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    for c in class_to_sum:
        fg = (labels == c).float()  # foreground for class c
        if (classes == 'present' and fg.sum() == 0):
            continue
        if C == 1:
            if len(classes) > 1:
                raise ValueError('Sigmoid output possible only with 1 class')
            class_pred = probas[:, 0]
        else:
            class_pred = probas[:, c]
        errors = (Variable(fg) - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, Variable(lovasz_grad(fg_sorted))))
    return mean(losses)


def flatten_probas(probas, labels, ignore=None):
    """
    Flattens predictions in the batch
    """
    if probas.dim() == 3:
        # assumes output of a sigmoid layer
        B, H, W = probas.size()
        probas = probas.view(B, 1, H, W)
    B, C, H, W = probas.size()
    probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
    labels = labels.view(-1)
    if ignore is None:
        return probas, labels
    valid = (labels != ignore)
    vprobas = probas[torch.squeeze(torch.nonzero(valid))]
    vlabels = labels[valid]
    return vprobas, vlabels


class Lovasz_softmax(torch.nn.Module):
    def __init__(self, classes='present', per_image=False, ignore=None):
        super(Lovasz_softmax, self).__init__()
        self.classes = classes
        self.per_image = per_image
        self.ignore = ignore

    def forward(self, probas, labels):
        return lovasz_softmax(probas, labels, self.classes, self.per_image, self.ignore)


class DiceLoss(_WeightedLoss):
    """
    This criterion is based on Dice coefficients.

    Modified version of: https://github.com/ai-med/nn-common-modules/blob/master/nn_common_modules/losses.py (MIT)
    Arxiv paper: https://arxiv.org/pdf/1606.04797.pdf
    """

    def __init__(
        self, 
        weight: Optional[torch.FloatTensor] = None,
        ignore_index: int = 255,
        binary: bool = False,
        reduction: str = 'mean'):
        """
        :param weight:  <torch.FloatTensor: n_class>. Optional scalar weight for each class.
        :param ignore_index: Label id to ignore when calculating loss.
        :param binary: Whether we are only doing binary segmentation.
        :param reduction: Specifies the reduction to apply to the output. Can be 'none', 'mean' or 'sum'.
        """
        super().__init__(weight=weight, reduction=reduction)
        self.ignore_index = ignore_index
        self.binary = binary

    def forward(self, predictions: torch.FloatTensor, targets: torch.LongTensor) -> torch.FloatTensor:
        """
        Forward pass.
        :param predictions: <torch.FloatTensor: n_samples, C, H, W>. Predicted scores.
        :param targets: <torch.LongTensor: n_samples, H, W>. Target labels.
        :return: <torch.FloatTensor: 1>. Scalar loss output.
        """
        self._check_dimensions(predictions=predictions, targets=targets)
        predictions = F.softmax(predictions, dim=1)
        if self.binary:
            return self._dice_loss_binary(predictions, targets)
        return self._dice_loss_multichannel(predictions, targets, self.weight, self.ignore_index)

    @staticmethod
    def _dice_loss_binary(predictions: torch.FloatTensor, targets: torch.LongTensor) -> torch.FloatTensor:
        """
        Dice loss for one channel binarized input.
        :param predictions: <torch.FloatTensor: n_samples, 1, H, W>. Predicted scores.
        :param targets: <torch.LongTensor: n_samples, H, W>. Target labels.
        :return: <torch.FloatTensor: 1>. Scalar loss output.
        """
        eps = 0.0001

        assert predictions.size(1) == 1, 'predictions should have a class size of 1 when doing binary dice loss.'

        intersection = predictions * targets

        # Summed over batch, height and width.
        numerator = 2 * intersection.sum(0).sum(1).sum(1)
        denominator = predictions + targets
        denominator = denominator.sum(0).sum(1).sum(1) + eps
        loss_per_channel = 1 - (numerator / denominator)

        # Averaged by classes.
        return loss_per_channel.sum() / predictions.size(1)

    @staticmethod
    def _dice_loss_multichannel(predictions: torch.FloatTensor,
                                targets: torch.LongTensor,
                                weight: Optional[torch.FloatTensor] = None,
                                ignore_index: int = -100) -> torch.FloatTensor:
        """
        Calculate the loss for multichannel predictions.
        :param predictions: <torch.FloatTensor: n_samples, n_class, H, W>. Predicted scores.
        :param targets: <torch.LongTensor: n_samples, H, W>. Target labels.
        :param weight:  <torch.FloatTensor: n_class>. Optional scalar weight for each class.
        :param ignore_index: Label id to ignore when calculating loss.
        :return: <torch.FloatTensor: 1>. Scalar loss output.
        """
        eps = 0.0001
        encoded_target = predictions.detach() * 0

        mask = targets == ignore_index
        targets = targets.clone()
        targets[mask] = 0
        encoded_target.scatter_(1, targets.unsqueeze(1), 1)
        mask = mask.unsqueeze(1).expand_as(encoded_target)
        encoded_target[mask] = 0

        intersection = predictions * encoded_target
        numerator = 2 * intersection.sum(0).sum(1).sum(1)
        denominator = predictions + encoded_target

        denominator[mask] = 0
        denominator = denominator.sum(0).sum(1).sum(1)
        if denominator.sum() == 0:
            # Means only void gradients. Summing the denominator would lead to loss of 0.
            return denominator.sum()
        denominator = denominator + eps

        if weight is None:
            weight = 1
        else:
            # We need to ensure that the weights and the loss terms resides in the same device id.
            # Especially crucial when we are using DataParallel/DistributedDataParallel.
            weight = weight / weight.mean()

        loss_per_channel = weight * (1 - (numerator / denominator))

        # Averaged by classes.
        return loss_per_channel.sum() / predictions.size(1)

    def _check_dimensions(self, predictions: torch.FloatTensor, targets: torch.LongTensor) -> None:
        error_message = ""
        if predictions.size(0) != targets.size(0):
            error_message += f'Predictions and targets should have the same batch size, but predictions have batch ' f'size {predictions.size(0)} and targets have batch size {targets.size(0)}.\n'
        if self.weight is not None and self.weight.size(0) != predictions.size(1):
            error_message += f'Weights and the second dimension of predictions should have the same dimensions ' f'equal to the number of classes, but weights has dimension {self.weight.size()} and ' f'targets has dimension {targets.size()}.\n'
        if self.binary and predictions.size(1) != 1:
            error_message += f'Binary class should have one channel representing the number of classes along the ' f'second dimension of the predictions, but the actual dimensions of the predictions ' f'is {predictions.size()}\n'
        if not self.binary and predictions.size(1) == 1:
            error_message += f'Predictions has dimension {predictions.size()}. The 2nd dimension equal to 1 ' f'indicates that this is binary, but binary was set to {self.binary} by construction\n'
        if error_message:
            raise ValueError(error_message)


class CrossEntropyDiceLoss(nn.Module):
    """ This is the combination of Cross Entropy and Dice Loss. """

    def __init__(self, reduction: str = 'mean', ignore_index: int = -100, weight: torch.Tensor = None):
        """
        :param reduction: Specifies the reduction to apply to the output. Can be 'none', 'mean' or 'sum'.
        :param ignore_index: Label id to ignore when calculating loss.
        :param weight:  <torch.FloatTensor: n_class>. Optional scalar weight for each class.
        """
        super(CrossEntropyDiceLoss, self).__init__()
        self.dice = DiceLoss(ignore_index=ignore_index, reduction=reduction, weight=weight)
        self.cross_entropy = nn.CrossEntropyLoss(weight=weight, reduction=reduction, ignore_index=ignore_index)

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor):
        """
        Calculate the loss.
        :param predictions: <torch.FloatTensor: n_samples, n_class, H, W>. Predicted scores.
        :param targets: <torch.LongTensor: n_samples, H, W>. Target labels.
        :return: <torch.FloatTensor: 1>. Scalar loss output.
        """
        ce_loss = self.cross_entropy(predictions, targets)
        dice_loss = self.dice(predictions, targets)
        return ce_loss + dice_loss


class BoundaryLoss(nn.Module):
    """
    Boundary Loss proposed in:
        Alexey Bokhovkin et al., Boundary Loss for Remote Sensing Imagery Semantic Segmentation
        https://arxiv.org/abs/1905.07852
    """

    def __init__(self, theta0=3, theta=5):
        super().__init__()

        self.theta0 = theta0
        self.theta = theta

    def forward(self, pred, gt):
        """
        Input:
            - pred: the output from model (before softmax), shape (N, C, H, W)
            - gt: ground truth map, shape (N, H, w)
        Return:
            - boundary loss, averaged over mini-bathc
        """
        n, c, _, _ = pred.shape

        # one-hot vector of ground truth
        one_hot_gt = one_hot(gt, c)

        # boundary map
        gt_b = F.max_pool2d(
            1 - one_hot_gt, kernel_size=self.theta0, stride=1, padding=(self.theta0 - 1) // 2)
        gt_b -= 1 - one_hot_gt


        pred_b = F.max_pool2d(
            1 - pred, kernel_size=self.theta0, stride=1, padding=(self.theta0 - 1) // 2)
        pred_b -= 1 - pred

        gt_b = gt_b.view(n, c, -1)
        pred_b = pred_b.view(n, c, -1)

        # Precision, Recall
        P = torch.sum(pred_b * gt_b, dim=2) / (torch.sum(pred_b, dim=2) + 1e-7)
        R = torch.sum(pred_b * gt_b, dim=2) / (torch.sum(gt_b, dim=2) + 1e-7)

        # Boundary F1 Score
        BF1 = 2 * P * R / (P + R + 1e-7)

        # summing BF1 Score for each class and average over mini-batch
        loss = torch.mean(1 - BF1)

        return loss


def one_hot(label, n_classes, requires_grad=True):
    """Return One Hot Label"""
    divce = label.device
    one_hot_label = torch.eye(
        n_classes, device=divce, requires_grad=requires_grad)[label]
    one_hot_label = one_hot_label.transpose(1, 3).transpose(2, 3)

    return one_hot_label
