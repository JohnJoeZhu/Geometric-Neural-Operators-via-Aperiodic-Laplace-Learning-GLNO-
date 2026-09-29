import torch
import numpy as np

#All loss function
def output_loss_(loss_record, meaning):
    output=""
    for i, loss in enumerate(loss_record):
        if meaning != None:
            output += "{}: ".format(meaning[i])
        output += "{:.6f} | ".format(loss)
    return output

class classify_nll_loss(object):
    def __init__(self, reduction='mean'):
        self.reduction = reduction

    def _compute_sample(self, preds, labels, num_classes):
        """
        计算单个样本的各项损失和指标。
        preds: [N, C] log-probabilities (after log_softmax)
        labels: [N] 类别索引
        返回: (loss, correct, num_valid, dice_grad, dice_numpy)
        """
        N = labels.shape[0]
        if N == 0:
            return 0.0, 0, 0, 0.0, 1.0

        loss = torch.nn.functional.nll_loss(preds, labels, reduction='mean')

        pred_labels = torch.max(preds, dim=-1).indices  # [N]
        correct = (pred_labels == labels).sum().item()

        dice_grad = self.dice_grad(preds, labels, num_classes)

        dice_numpy = self.dice(pred_labels, labels, num_classes)

        return loss, correct/N, dice_grad, dice_numpy

    def __call__(self, preds, labels, mask=None, **kwargs):
        """
        preds: 若 mask=None，形状为 [N, C]；否则为 [B, T, C]
        labels: 若 mask=None，形状为 [N]；否则为 [B, T]
        mask: [B, T] 布尔张量，True 表示有效点。若为 None，视为单样本处理。
        返回列表: [0.3*dice_grad + loss, loss, -accuracy, dice_grad, -dice_numpy]
        """
        if mask is None:
            # 单样本模式（兼容原逻辑）
            loss, acc, dice_grad, dice_numpy = self._compute_sample(preds, labels, preds.shape[-1])
            return [0.3 * dice_grad + loss, loss, -acc, dice_grad, -dice_numpy]
        else:
            # 批处理模式：按样本分别计算
            batch_size = preds.shape[0]
            num_classes = preds.shape[-1]

            total_loss = 0.0
            total_dice_grad = 0.0
            total_dice_numpy = 0.0
            total_acc = 0

            for i in range(batch_size):
                # 提取当前样本的有效点
                valid_mask = mask[i].squeeze(-1)  # [T]
                # print(valid_mask.shape)
                # print(preds.shape)
                # print(labels.shape)
                if not valid_mask.any():
                    # 若该样本无有效点，跳过（或按空样本处理）
                    continue
                preds_i = preds[i, valid_mask]      # [num_valid_i, C]
                labels_i = labels[i, valid_mask]    # [num_valid_i]

                loss_i, acc_i, dice_grad_i, dice_numpy_i = self._compute_sample(
                    preds_i, labels_i, num_classes
                )

                total_loss += loss_i
                total_dice_grad += dice_grad_i
                total_dice_numpy += dice_numpy_i
                total_acc += acc_i

            return [0.3 * total_dice_grad + total_loss, total_loss, -total_acc, total_dice_grad, -total_dice_numpy]

    def dice_grad(self, preds, targets, num_classes, smooth=1e-6):
        probs = torch.exp(preds)                 # [N, C]
        targets = targets.contiguous().view(-1)
        probs = probs.contiguous()
        targets_one_hot = torch.zeros_like(probs).scatter_(1, targets.unsqueeze(1), 1)
        intersection = (probs * targets_one_hot).sum(dim=0)   # [C]
        union = probs.sum(dim=0) + targets_one_hot.sum(dim=0) # [C]
        dice_per_class = (2.0 * intersection + smooth) / (union + smooth)
        loss = 1 - dice_per_class.mean()
        return loss

    def dice(self, preds, labels, num_class):
        preds = preds.cpu().detach().numpy()
        labels = labels.cpu().detach().numpy()
        dice = np.ones(num_class)
        for i in range(0, num_class):
            labels_indices = np.where(labels == i)[0]
            preds_indices = np.where(preds == i)[0]
            if len(labels_indices) + len(preds_indices) > 0:
                dice[i] = 2 * len(np.intersect1d(preds_indices, labels_indices)) / (len(preds_indices) + len(labels_indices))
        return np.mean(dice)

class classify_nll_loss_global(object): 
    def __init__(self, reduction='mean'):
        self.reduction = reduction

    def __call__(self, preds, labels, **kwargs):
        # print(preds.shape,labels.shape)
        # loss = torch.nn.functional.cross_entropy(pred, target, reduction=self.reduction)
        # print(preds,labels)
        loss=torch.nn.functional.nll_loss(preds,labels)
        pred_labels = torch.max(preds, dim=-1).indices
        this_correct = pred_labels.eq(labels).sum().item()
        # print(this_correct)
        # dice=self.dice(pred_labels,labels,preds.shape[-1])
        return [loss,-this_correct]
    
class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True): 
        super(LpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        all_norms = torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)
        
        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y, mask=None):
        if mask is not None:
            mask=mask.to(x.device)
            # print(x.shape,y.shape,mask.shape)
            x=x*mask
            y=y*mask
        num_examples = x.shape[0]

        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)
        assert torch.isnan(y_norms).any() == False, "y_norms contains NaN"
        assert torch.isnan(diff_norms).any() == False, "diff_norms contains NaN"
        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def __call__(self, x, y, **kwargs):
        return self.rel(x, y, **kwargs)

class CarCFDLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(CarCFDLoss, self).__init__()

        self.lp_loss = LpLoss(d=d, p=p, size_average=size_average, reduction=reduction)

    def compute_loss(self, x, y, sep=True, mask=None, **kwargs):
        if mask is not None:
            mask=mask.to(x.device)
            x=x*mask
            y=y*mask
        if len(x.shape)==2:
            x=x.unsqueeze(0)
            y=y.unsqueeze(0)
        press_loss = self.lp_loss(x[:, :, -1], y[:, :, -1])
        vol_loss = self.lp_loss(x[:, :, :-1], y[:, :, :-1])
        
        if sep:
            return [press_loss + vol_loss, press_loss, vol_loss]
        else:
            return press_loss + vol_loss
    
    def __call__(self, x, y, **kwargs):
        return self.compute_loss(x, y, **kwargs)


class CFlowLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(CFlowLoss, self).__init__()

        self.lp_loss = LpLoss(d=d, p=p, size_average=size_average, reduction=reduction)

    def compute_loss(self, x, y, sep=True, mask=None, **kwargs):
        if mask is not None:
            mask=mask.to(x.device)
            x=x*mask
            y=y*mask
        if len(x.shape)==2:
            x=x.unsqueeze(0)
            y=y.unsqueeze(0)
        press_loss = self.lp_loss(x[:, :, 0], y[:, :, 0])
        vol_loss = self.lp_loss(x[:, :, 1:], y[:, :, 1:])
        vx_loss=self.lp_loss(x[:, :, 1], y[:, :, 1])
        vy_loss=self.lp_loss(x[:, :, 2], y[:, :, 2])
        
        if sep:
            return [press_loss + vol_loss, press_loss, vol_loss, vx_loss, vy_loss]
        else:
            return press_loss + vol_loss
    
    def __call__(self, x, y, **kwargs):
        return self.compute_loss(x, y, **kwargs)

class TurbulentLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(TurbulentLoss, self).__init__()

        self.lp_loss = LpLoss(d=d, p=p, size_average=size_average, reduction=reduction)

    def compute_loss(self, x, y, sep=True, **kwargs):
        dens_loss = self.lp_loss(x[:, :, 0], y[:, :, 0])
        pres_loss = self.lp_loss(x[:, :, 1], y[:, :, 1])
        vol_loss = self.lp_loss(x[:, :, 2:], y[:, :, 2:])
        vx_loss=self.lp_loss(x[:, :, 2], y[:, :, 2])
        vy_loss=self.lp_loss(x[:, :, 3], y[:, :, 3])
        
        if sep:
            return [dens_loss + pres_loss + vol_loss, dens_loss, pres_loss, vol_loss, vx_loss, vy_loss]
        else:
            return dens_loss + pres_loss + vol_loss
    
    def __call__(self, x, y, **kwargs):
        return self.compute_loss(x, y, **kwargs)

class MultipleLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        
        self.lp_loss = LpLoss(d=d, p=p, size_average=size_average, reduction=reduction)
    
    def compute_loss(self, x, y, sep=True, mask=None, **kwargs):
        if mask is not None:
            mask=mask.to(x.device)
            x=x*mask
            y=y*mask
        num_feature = x.size(2)
        loss_list = []
        for i in range(num_feature):
            loss_list.append(self.lp_loss(x[:, :, i], y[:, :, i]))
        
        all_loss = sum(loss_list)
        
        if sep:
            return [all_loss] + loss_list
        else:
            return all_loss
    
    def __call__(self, x, y, **kwargs):
        return self.compute_loss(x, y, **kwargs)

def label_smoothing_log_loss(pred, labels, smoothing=0.0):
    n_class = pred.shape[-1]
    one_hot = torch.zeros_like(pred)
    one_hot[labels] = 1.
    one_hot = one_hot * (1 - smoothing) + (1 - one_hot) * smoothing / (n_class - 1)
    loss = -(one_hot * pred).sum(dim=-1).mean()
    return loss