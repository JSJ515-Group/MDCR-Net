import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ParsingRelationLoss(nn.Module):
    def __init__(self):
        super(ParsingRelationLoss, self).__init__()

    def forward(self, logits):
        n, c, h, w = logits.shape
        loss_all = []
        # 计算相邻的两点之间的差异
        for i in range(0, h - 1):
            loss_all.append(logits[:, :, i, :] - logits[:, :, i + 1, :])
        # loss0 : n,c,w
        loss = torch.cat(loss_all)
        return torch.nn.functional.smooth_l1_loss(loss, torch.zeros_like(loss))


# 二阶约束差分方程(约束车道线形状)
class ParsingRelationDis(nn.Module):
    def __init__(self):
        super(ParsingRelationDis, self).__init__()
        self.l1 = torch.nn.L1Loss()
        # self.l1 = torch.nn.MSELoss()

    def forward(self, x):
        n, dim, num_rows, num_cols = x.shape
        # 对得到的预测进行softmax,并且去除最后一个维度
        x = torch.nn.functional.softmax(x[:, :dim - 1, :, :], dim=1)
        embedding = torch.Tensor(np.arange(dim - 1)).float().to(x.device).view(1, -1, 1, 1)
        pos = torch.sum(x * embedding, dim=1)

        diff_list1 = []
        for i in range(0, num_rows // 2):
            diff_list1.append(pos[:, i, :] - pos[:, i + 1, :])

        loss = 0
        for i in range(len(diff_list1) - 1):
            loss += self.l1(diff_list1[i], diff_list1[i + 1])
        loss /= len(diff_list1) - 1
        return loss


class OhemCELoss(nn.Module):
    def __init__(self, thresh, n_min, ignore_lb=255, *args, **kwargs):
        super(OhemCELoss, self).__init__()
        # 参数thresh表示阈值，n_min表示最小样本数，ignore_lb表示忽略的标签值，默认为255，*args和**kwargs为其他参数
        # 将阈值转换为对数形式，并将其放到GPU上
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float)).cuda()
        self.n_min = n_min
        self.ignore_lb = ignore_lb
        # 创建一个交叉熵损失函数实例，设置忽略标签和输出不进行降维处理
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb, reduction='none')

    def forward(self, logits, labels):
        N, C, H, W = logits.size()
        loss = self.criteria(logits, labels).view(-1)
        # 计算交叉熵损失，并将其展平为一维向量
        loss, _ = torch.sort(loss, descending=True)
        # 对损失进行降序排序
        if loss[self.n_min] > self.thresh:
            # 如果第n_min个位置的损失大于阈值
            loss = loss[loss > self.thresh]
            # 将损失中大于阈值的部分保留
        else:
            loss = loss[:self.n_min]
            # 如果第n_min个位置的损失小于等于阈值，只保留前n_min个样本的损失

        return torch.mean(loss)
        # 返回保留样本损失的均值作为最终的损失值


def update_threshold(threshold, loss, alpha=0.5, min_threshold=0.0, max_threshold=1.0):
    """
    根据当前损失值动态更新阈值
    :param threshold: 当前阈值
    :param loss: 当前批次的损失值
    :param alpha: 阈值更新系数
    :param min_threshold: 最小阈值
    :param max_threshold: 最大阈值
    :return: 更新后的阈值
    """
    # 计算新的阈值
    new_threshold = threshold + alpha * (loss - threshold)

    # 对新阈值进行边界处理
    new_threshold = max(min_threshold, min(max_threshold, new_threshold))

    # 返回更新后的阈值
    return new_threshold


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, input, target):
        # 将输入和目标转换为二进制形式，即预测概率大于0.5为正类，否则为负类
        input = (input > 0.5).float()
        target = (target > 0.5).float()
        intersection = torch.sum(input * target)  # 计算交集
        union = torch.sum(input) + torch.sum(target)  # 计算并集
        dice_score = (2. * intersection + self.smooth) / (union + self.smooth)  # 计算 Dice 系数
        loss = 1 - dice_score  # 将 Dice 系数转换为损失函数
        return loss


def dice_loss(input, target):
    input = input.contiguous().view(input.size()[0], -1)
    target = target.contiguous().view(target.size()[0], -1).float()
    a = torch.sum(input * target, 1)
    b = torch.sum(input * input, 1) + 0.001
    c = torch.sum(target * target, 1) + 0.001
    d = (2 * a) / (b + c)
    return (1 - d).mean()
