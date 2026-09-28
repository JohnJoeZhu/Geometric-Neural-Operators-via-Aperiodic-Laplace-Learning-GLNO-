import os
import torch
import torch.distributed as dist
import numpy as np
import logging
from datetime import datetime
import yaml

# == Pytorch things

def __concat_collate_fn(batch):
    """自定义collate函数，在特征维度拼接evecs"""
    concatenated_batch = {}
    
    # 获取batch大小
    batch_size = len(batch)
    
    # 处理evecs - 在特征维度拼接
    evecs_tensors = [sample["evecs"] for sample in batch]  # 每个都是 [num_point, 128]
    
    # 计算每个样本的点数
    num_points_list = [tensor.shape[0] for tensor in evecs_tensors]
    total_points = sum(num_points_list)
    
    # 创建全零的大矩阵 [sum_num_points, 128*batch_size]
    concatenated_evecs = torch.zeros(total_points, 128 * batch_size)
    
    # 将每个样本的evecs放到对应的位置
    start_idx = 0
    for i, (tensor, num_points) in enumerate(zip(evecs_tensors, num_points_list)):
        feature_start = i * 128
        feature_end = (i + 1) * 128
        concatenated_evecs[start_idx:start_idx+num_points, feature_start:feature_end] = tensor
        start_idx += num_points
    
    # 添加batch维度 [1, sum_num_points, 128*batch_size]
    concatenated_batch["evecs"] = concatenated_evecs.unsqueeze(0)
    
    # 其他字段仍然在点数维度拼接
    for key in batch[0].keys():
        if key != "evecs":
            tensors = [sample[key] for sample in batch]
            concatenated = torch.cat(tensors, dim=0)  # [sum_num_points, ...]
            concatenated_batch[key] = concatenated.unsqueeze(0)  # [1, sum_num_points, ...]
    
    return concatenated_batch
    

def concat_collate_fn_globa_label(batch):
    """
    自定义collate函数：
    - evecs: 在特征维度拼接，形状变为 [1, total_points, 128 * batch_size]
    - 其他点级别字段（mass, point_labels等）在点数维度拼接
    - 添加 batch_idx 字段，记录每个点所属的样本索引
    - 样本级别字段（如 global_label）直接堆叠成 [batch_size] 张量
    """
    batch_size = len(batch)
    num_points_list = [sample['evecs'].shape[0] for sample in batch]
    total_points = sum(num_points_list)

    collated = {}

    # ---------- 生成 batch_idx ----------
    batch_idx = torch.zeros(total_points, dtype=torch.long)
    start = 0
    for i, n in enumerate(num_points_list):
        batch_idx[start:start + n] = i
        start += n
    collated['batch_idx'] = batch_idx.unsqueeze(0)  # [1, total_points]

    # ---------- 处理 evecs (特征维度拼接) ----------
    evecs_concat = torch.zeros(total_points, 128 * batch_size)
    start = 0
    for i, (n, sample) in enumerate(zip(num_points_list, batch)):
        tensor = sample['evecs']  # [n, 128]
        evecs_concat[start:start + n, i*128:(i+1)*128] = tensor
        start += n
    collated['evecs'] = evecs_concat.unsqueeze(0)  # [1, total_points, 128*batch_size]

    # ---------- 处理其他点级别字段（如 mass, point_labels） ----------
    point_fields = ['mass','evals','geo_feat','vertices','faces']  # 根据你的实际字段名修改
    for field in point_fields:
        if field in batch[0]:
            tensors = [sample[field] for sample in batch]
            # 假设 mass 是 [n] 的一维张量，直接拼接
            concatenated = torch.cat(tensors, dim=0)  # [total_points]
            collated[field] = concatenated.unsqueeze(0)  # [1, total_points]

    # ---------- 处理样本级别字段（如全局标签） ----------
    sample_fields = ['labels']  # 根据你的实际字段名修改
    for field in sample_fields:
        if field in batch[0]:
            values = [sample[field] for sample in batch]
            collated[field] = torch.tensor(values)  # [batch_size]
    return collated

def pad_collate_fn_cflow(batch):
    """
    自定义collate函数，将每个样本的点云填充到最大点数，然后堆叠成批量。
    返回的字典包含：
        - evecs: [batch_size, max_points, 128]  (或对应特征维度)
        - mass: [batch_size, max_points]
        - evals: [batch_size, 128]  (假设为 [n, k] 形状)
        - geo_feat: [batch_size, max_points, ...]
        - vertices: [batch_size, max_points, 3]  (假设为 [n, 3])
        - labels: [batch_size, max_points]  (每个点的标签)
        - input: [batch_size, max_points, ...] (根据实际形状)
        - mask: [batch_size, max_points]  (布尔张量，True表示有效点)
        - 样本级别字段（如 global_label）: [batch_size] 堆叠
    """
    batch_size = len(batch)
    if batch_size == 1:
        return batch[0]
    
    num_points_list = [sample['evecs'].shape[0] for sample in batch]
    max_points = max(num_points_list)
    
    collated = {'evals':torch.stack([sample['evals'] for sample in batch])}
    
    point_fields = ['evecs', 'mass', 'geo_feat', 'vertices', 'labels', 'input']

    shapes = {}
    for field in point_fields:
        if field in batch[0]:
            sample_tensor = batch[0][field]
            shapes[field] = sample_tensor.shape[1:] 

    for field, feat_shape in shapes.items():
        # 创建 [batch_size, max_points, *feat_shape] 的零张量
        full_shape = (batch_size, max_points) + feat_shape
        collated[field] = torch.zeros(full_shape, dtype=batch[0][field].dtype)
    
    mask = torch.zeros(batch_size, max_points, 1, dtype=torch.bool)
    
    for i, sample in enumerate(batch):
        n = num_points_list[i]
        for field, feat_shape in shapes.items():
            tensor = sample[field]  # [n, *feat_shape]
            collated[field][i, :n, ...] = tensor
        mask[i, :n, 0] = 1
    
    collated['mask'] = mask
    
    # # ---------- 处理样本级别字段（如 global_label）----------
    # # 假设这些字段在 batch 中所有样本都有，且不是点级别
    # sample_fields = ['global_label']  # 根据实际情况扩展
    # for field in sample_fields:
    #     if field in batch[0]:
    #         collated[field] = torch.tensor([sample[field] for sample in batch])
    
    return collated

from torch.nn.utils.rnn import pad_sequence

def pad_collate_fn_cflow_optimized(batch):
    batch_size = len(batch)

    # 全局字段直接 stack
    collated = {
        'evals': torch.stack([sample['evals'] for sample in batch])
    }

    # 点级字段，每个字段是一个列表，每个元素为 [n_i, *feat_shape]
    point_fields = ['evecs', 'mass', 'geo_feat', 'vertices', 'labels', 'input']
    for field in point_fields:
        if field not in batch[0]:
            continue
        # 提取该字段的所有样本张量
        tensors = [sample[field] for sample in batch]
        # pad_sequence 自动填充到最长序列，padding_value=0
        # 返回形状 [max_points, batch_size, *feat_shape]
        padded = pad_sequence(tensors, batch_first=False, padding_value=0)
        # 转为 batch_first 并添加到 collated
        collated[field] = padded.permute(1, 0, *range(2, padded.dim()))

    # 创建 mask：有效位置为 True
    num_points_list = [sample['evecs'].shape[0] for sample in batch]
    max_points = max(num_points_list)
    mask = torch.zeros(batch_size, max_points, dtype=torch.bool)
    for i, n in enumerate(num_points_list):
        mask[i, :n] = True
    collated['mask'] = mask.unsqueeze(-1)  # 保持与原来一致 [B, max_points, 1]

    return collated

def pad_collate_fn_intra(batch):
    """
    自定义collate函数，将每个样本的点云填充到最大点数，然后堆叠成批量。
    返回的字典包含：
        - evecs: [batch_size, max_points, 128]  (或对应特征维度)
        - mass: [batch_size, max_points]
        - evals: [batch_size, 128]  (假设为 [n, k] 形状)
        - geo_feat: [batch_size, max_points, ...]
        - vertices: [batch_size, max_points, 3]  (假设为 [n, 3])
        - labels: [batch_size, max_points]  (每个点的标签)
        - input: [batch_size, max_points, ...] (根据实际形状)
        - mask: [batch_size, max_points]  (布尔张量，True表示有效点)
        - 样本级别字段（如 global_label）: [batch_size] 堆叠
    """
    batch_size = len(batch)
    # print(batch_size)
    # 获取每个样本的点数
    num_points_list = [sample['evecs'].shape[0] for sample in batch]
    max_points = max(num_points_list)
    
    collated = {'evals':torch.stack([sample['evals'] for sample in batch])}
    # print(collated['evals'].shape)
    # ---------- 确定所有可能存在的点级别字段 ----------
    point_fields = ['evecs', 'mass', 'geo_feat', 'vertices']
    # 对于每个字段，我们需要知道其每个点的特征维度（除了点数维度）
    # 假设 evecs 是 [n, 128]；mass 是 [n]；evals 可能是 [n, k]；geo_feat 类似；vertices 是 [n, 3]；labels 是 [n]；input 是 [n, ...]
    # 我们根据第一个样本的形状来推断，并假设同一字段在所有样本中形状一致（除了点数）
    shapes = {}
    for field in point_fields:
        if field in batch[0]:
            sample_tensor = batch[0][field]
            # 去掉第一个维度（点数），保留其余维度
            shapes[field] = sample_tensor.shape[1:]  # 可能为空（标量每个点）
    # print(shapes)
    # ---------- 初始化填充后的张量 ----------
    for field, feat_shape in shapes.items():
        # 创建 [batch_size, max_points, *feat_shape] 的零张量
        full_shape = (batch_size, max_points) + feat_shape
        collated[field] = torch.zeros(full_shape, dtype=batch[0][field].dtype)
    
    # ---------- 创建 mask ----------
    mask = torch.zeros(batch_size, max_points, 1, dtype=torch.bool)
    
    # ---------- 填充数据 ----------
    for i, sample in enumerate(batch):
        n = num_points_list[i]
        for field, feat_shape in shapes.items():
            tensor = sample[field]  # [n, *feat_shape]
            # 将前n行填入对应位置
            # print(field,collated[field][i, :n, ...].shape,tensor.shape)
            collated[field][i, :n, ...] = tensor
        mask[i, :n, 0] = 1
    
    collated['mask'] = mask
    
    # # ---------- 处理样本级别字段（如 global_label）----------
    # # 假设这些字段在 batch 中所有样本都有，且不是点级别
    sample_fields = ['labels']  # 根据实际情况扩展
    for field in sample_fields:
        if field in batch[0]:
            collated[field] = torch.tensor([sample[field] for sample in batch])
    return collated

COLLATE_Path={
    'cylinder_flow': pad_collate_fn_cflow_optimized,
    'intra': pad_collate_fn_cflow,
    'cortex': pad_collate_fn_cflow,
    'car': pad_collate_fn_cflow,
    'poisson': pad_collate_fn_cflow,
    'poissonunstruc': pad_collate_fn_cflow,
}


import math
class HalfCosineAnnealingLR:
    def __init__(self, optimizer, T_max, eta_min=0):
        self.optimizer = optimizer
        self.T_max = T_max
        self.eta_min = eta_min
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.last_epoch = -1

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch

        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            if epoch < self.T_max:
                # 余弦下降（半个周期）
                cos = math.cos(math.pi * epoch / self.T_max)
                lr = self.eta_min + (base_lr - self.eta_min) * (1 + cos) / 2
            else:
                # 超过 T_max 后保持最低
                lr = self.eta_min
            param_group['lr'] = lr

def build_scheduler(optimizer, args):
    if args['name'] == 'MultiStepLR':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=args['milestones'],
            gamma=args['gamma'],
        )
    elif args['name'] == 'OneCycleLR':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args['lr'],
            div_factor=args['div_factor'],
            final_div_factor=args['final_div_factor'],
            pct_start=args['pct_start'],
            steps_per_epoch=args['steps_per_epoch'],
            epochs=args['epochs'],
        )
    elif args['name'] == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=args['step_size'],
            gamma=args['gamma'],
        )
    elif args['name'] == 'CosineAnnealingLR':
        scheduler = HalfCosineAnnealingLR(
            optimizer, 
            T_max=args['T_max'], 
            eta_min=args['eta_min']
        )
    elif args['name'] == 'CosineAnnealingWarmRestarts':
        scheduler=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=args['T_0'], 
            T_mult=args['T_mult'],
        )
    elif args['name'] == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=args['gamma'], patience=args['patience'])
    elif args['name'] is None:
        scheduler = None
    else:
        raise NotImplementedError("Scheduler {} not implemented".format(args['name']))
    return scheduler

def split_train_val(len,ratio):
    indices = np.arange(len)
    np.random.shuffle(indices)
    split = int(len * (1 - ratio))
    train_indices = indices[:split].tolist()
    val_indices = indices[split:].tolist()
    return train_indices, val_indices

def convert_label_to_one_hot(labels, num_classes=None):
    """
    将标签张量转换为 one-hot 编码
    
    参数:
    labels: 形状为 [b, n] 的整数张量，每个元素表示类别索引
    num_classes: 类别数量，如果不提供则使用 labels 中的最大值+1
    
    返回:
    one_hot: 形状为 [b, n, c] 的 one-hot 编码张量
    """
    # 获取输入形状
    b, n = labels.shape
    
    # 确定类别数量
    if num_classes is None:
        num_classes = labels.max().item() + 1
    
    # 创建 one-hot 编码张量
    one_hot = torch.zeros(b, n, num_classes, device=labels.device, dtype=torch.float32)
    
    # 使用 scatter_ 函数将对应位置设置为 1
    # 注意: scatter_ 要求索引与目标张量维度匹配，所以需要扩展 labels 的维度
    one_hot.scatter_(2, labels.unsqueeze(-1), 1)
    
    return one_hot


class ComplexGradientMonitor:
    def __init__(self, model, mode='magnitude', log_fn=print, log_interval=10):
        """
        mode: 'magnitude' 记录模长的均值/最值
              'real_imag' 分别记录实部和虚部的均值/最值
              'both'      同时记录模长和实部虚部
        log_fn: 输出函数，可以是 print 或 logger.info 等
        log_interval: 每隔多少步输出一次（需在训练循环中调用 log_step）
        """
        self.model = model
        self.mode = mode
        self.log_fn = log_fn
        self.log_interval = log_interval
        self.step_counter = 0   # 用于控制输出频率
        # 存储每个参数的当前梯度统计量，格式：{name: {...}}
        self.current_stats = {}

        # 注册钩子
        self.handles = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                handle = param.register_hook(self._make_hook(name))
                self.handles.append(handle)

    def _make_hook(self, name):
        def hook(grad):
            # grad 是梯度张量，可能是复数
            stats = {}

            if grad.is_complex():
                # 复数梯度处理
                if self.mode in ('magnitude', 'both'):
                    mag = grad.abs()  # 模长张量
                    stats['mag_mean'] = mag.mean().item()
                    stats['mag_max'] = mag.max().item()
                    stats['mag_min'] = mag.min().item()
                if self.mode in ('real_imag', 'both'):
                    real = grad.real.abs()
                    imag = grad.imag.abs()
                    stats['real_mean'] = real.mean().item()
                    stats['real_max'] = real.max().item()
                    stats['real_min'] = real.min().item()
                    stats['imag_mean'] = imag.mean().item()
                    stats['imag_max'] = imag.max().item()
                    stats['imag_min'] = imag.min().item()
            else:
                grad=grad.abs()
                # 实数梯度，保持原有统计
                stats['mean'] = grad.mean().item()
                stats['max'] = grad.max().item()
                stats['min'] = grad.min().item()
                # 如果需要绝对值均值，可添加
                stats['abs_mean'] = grad.abs().mean().item()

            self.current_stats[name] = stats
        return hook

    def log_step(self, force=False):
        """在训练循环的每个 step 调用，根据步数决定是否输出"""
        self.step_counter += 1
        if force or (self.step_counter % self.log_interval == 0):
            self._output_stats()

    def _output_stats(self):
        """将当前梯度统计量输出到 log_fn"""
        if not self.current_stats:
            return
        # 可以输出整体汇总，也可以逐个参数输出
        # 这里输出一个精简的汇总，便于观察
        # summary = []
        # for name, stats in self.current_stats.items():
        #     # 提取关键信息：对于复数，展示模长均值；对于实数，展示绝对值均值
        #     if 'mag_mean' in stats:
        #         key_val = f"mag_mean={stats['mag_mean']:.4e}"
        #     elif 'abs_mean' in stats:
        #         key_val = f"abs_mean={stats['abs_mean']:.4e}"
        #     elif 'mean' in stats:
        #         key_val = f"mean={stats['mean']:.4e}"
        #     else:
        #         key_val = ""
        #     summary.append(f"{name}: {key_val}\n")
        # self.log_fn(f"Step {self.step_counter} gradient stats: " + " | ".join(summary))

        # 如果需要详细输出所有统计量，可以扩展：
        for name, stats in self.current_stats.items():
            for key,val in stats.items():
                if val>1e-4:
                    continue
                # if 'mlp' in name:
                #     continue
                self.log_fn(f"{name}.{key}={val:.4e}")

    def remove_hooks(self):
        for handle in self.handles:
            handle.remove()