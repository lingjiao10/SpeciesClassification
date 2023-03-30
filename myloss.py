import torch
import torch.nn as nn
import torch.nn.functional as F


class BCELoss_class_weighted(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = weight # 二分类中正负样本的权重，第一项为负类权重，第二项为正类权重

    def forward(self, input, target):
        input = torch.clamp(input,min=1e-7,max=1-1e-7)
        bce = - self.weight[1] * target * torch.log(input) - (1 - target) * self.weight[0] * torch.log(1 - input)
        return torch.mean(bce)


class BCEWithSoftmaxLoss(nn.Module):
    def __init__(self, weight=None, reduction='mean'):
        super().__init__()
        self.weight = weight
        self.reduction = reduction
        # self.bce = 

    def forward(self, input, target):
        probs = F.softmax(input, dim=1)
        loss = F.binary_cross_entropy(probs, target, weight=self.weight, reduction=self.reduction)
        n = input.size()[0] #batch size
        if self.reduction == 'sum':
            return loss / (2*n)
        return loss

        # none 得到原始loss矩阵

class BCEWithSoftmaxFocalLoss(nn.Module):
    def __init__(self, gamma=1.5, alpha=0.25, weight=None, reduction='mean'):
        super().__init__()
        self.weight = weight
        self.reduction = reduction
        self.gamma = gamma
        self.alpha = alpha
        # self.bce = 

    def forward(self, input, target):
        probs = F.softmax(input, dim=1)
        loss = F.binary_cross_entropy(probs, target, weight=self.weight, reduction='none')

        # pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = target * probs + (1 - target) * (1 - probs)
        alpha_factor = target * self.alpha + (1 - target) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            n = input.size()[0] #batch size
            return loss.sum() / (2*n)
        else:  # 'none'
            return loss


        # n = input.size()[0] #batch size
        # if self.reduction == 'sum':
        #     return loss / (2*n)
        # return loss