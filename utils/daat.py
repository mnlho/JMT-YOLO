"""
DAAT: Distribution-Aware Adaptive Threshold
用于伪标签筛选的自适应阈值模块

核心思想：基于 Otsu 方法，自动从置信度分布中找到最优分割点，
无需手动设定阈值，完全数据驱动。

Usage:
    from utils.daat import DAAT
    
    # 初始化
    daat = DAAT(
        init_threshold=0.5,
        ema_momentum=0.9,
        lower_bound=0.3,
        upper_bound=0.9,
        num_bins=100
    )
    
    # 在 NMS 之前获取自适应阈值
    adaptive_threshold = daat.get_threshold(raw_predictions)
    
    # NMS 使用自适应阈值
    pred_nms = non_max_suppression(pred, conf_thres=adaptive_threshold, ...)
"""

import numpy as np
import torch
from typing import Optional, Tuple, List


class DAAT:
    """
    Distribution-Aware Adaptive Threshold (DAAT)
    
    基于 Otsu 方法的自适应阈值选择器，用于伪标签筛选。
    通过分析置信度分布的双峰特性，自动找到最优分割阈值。
    
    Args:
        init_threshold: 初始阈值（第一个 batch 使用）
        ema_momentum: EMA 动量，用于平滑阈值变化（0-1，越大越平滑）
        lower_bound: 阈值下界，防止阈值过低导致噪声标签
        upper_bound: 阈值上界，防止阈值过高导致标签过少
        num_bins: Otsu 直方图的 bin 数量
        warmup_batches: warmup 期间使用固定阈值的 batch 数
    """
    
    def __init__(
        self,
        init_threshold: float = 0.5,
        ema_momentum: float = 0.9,
        lower_bound: float = 0.25,
        upper_bound: float = 0.9,
        num_bins: int = 100,
        warmup_batches: int = 100,
        conf_filter_min: float = 0.5
    ):
        self.init_threshold = init_threshold
        self.ema_momentum = ema_momentum
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.num_bins = num_bins
        self.warmup_batches = warmup_batches
        self.conf_filter_min = conf_filter_min
        
        # 状态变量
        self.current_threshold = init_threshold
        self.ema_threshold = init_threshold
        self.batch_count = 0
        
        # 统计信息（用于可视化）
        self.last_otsu_threshold = None
        self.last_confidence_scores = None
        self.threshold_history = []
        
    def get_threshold(
        self, 
        predictions: torch.Tensor,
        return_stats: bool = False
    ) -> float:
        """
        根据当前 batch 的预测结果计算自适应阈值
        
        Args:
            predictions: 模型原始预测，shape: (batch, num_detections, 5+num_classes)
                        或者已经是置信度分数 tensor
            return_stats: 是否返回统计信息
            
        Returns:
            adaptive_threshold: 自适应阈值
            stats (optional): 统计信息字典
        """
        self.batch_count += 1
        
        # 提取最终置信度分数（已按 conf_filter_min 过滤）
        confidence_scores = self._extract_confidence(predictions)
        
        if confidence_scores is None or len(confidence_scores) < 10:
            # 置信度样本太少，使用当前阈值
            if return_stats:
                return self.current_threshold, {'reason': 'insufficient_samples'}
            return self.current_threshold
        
        self.last_confidence_scores = confidence_scores
        
        # Warmup 期间使用固定阈值
        if self.batch_count <= self.warmup_batches:
            if return_stats:
                return self.init_threshold, {'reason': 'warmup', 'batch': self.batch_count}
            return self.init_threshold
        
        # 使用 Otsu 方法计算最优阈值
        otsu_threshold = self._otsu_threshold(confidence_scores)
        self.last_otsu_threshold = otsu_threshold
        
        # 应用边界约束
        bounded_threshold = np.clip(otsu_threshold, self.lower_bound, self.upper_bound)
        
        # EMA 平滑
        self.ema_threshold = (
            self.ema_momentum * self.ema_threshold + 
            (1 - self.ema_momentum) * bounded_threshold
        )
        
        self.current_threshold = self.ema_threshold
        self.threshold_history.append(self.current_threshold)
        
        if return_stats:
            stats = {
                'otsu_raw': otsu_threshold,
                'bounded': bounded_threshold,
                'ema': self.ema_threshold,
                'num_samples': len(confidence_scores),
                'mean_conf': float(np.mean(confidence_scores)),
                'std_conf': float(np.std(confidence_scores))
            }
            return self.current_threshold, stats
        
        return self.current_threshold
    
    def _extract_confidence(self, predictions: torch.Tensor) -> Optional[np.ndarray]:
        """
        从预测结果中提取置信度分数
        
        支持多种输入格式：
        1. 原始预测 tensor (batch, num_det, 5+nc)
        2. NMS 后的结果 list
        3. 直接的置信度 tensor
        """
        if predictions is None:
            return None
            
        if isinstance(predictions, list):
            # NMS 后的结果格式：list of (n, 6) tensors
            all_confs = []
            for pred in predictions:
                if pred is not None and len(pred) > 0:
                    if isinstance(pred, torch.Tensor):
                        confs = pred[:, 4].detach().cpu().numpy()
                    else:
                        confs = pred[:, 4]
                    if self.conf_filter_min is not None:
                        confs = confs[confs > self.conf_filter_min]
                    all_confs.extend(confs.flatten().tolist())
            if len(all_confs) == 0:
                return None
            return np.array(all_confs)
        
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu()
            
            # 检查维度
            if predictions.dim() == 1:
                # 已经是置信度分数
                confs = predictions.numpy()
                if self.conf_filter_min is not None:
                    confs = confs[confs > self.conf_filter_min]
                return confs
            elif predictions.dim() == 2:
                # (num_det, features)
                # 最终置信度 = objectness * max(class_conf)
                if predictions.shape[1] >= 6:
                    objectness = predictions[:, 4]
                    class_conf = predictions[:, 5:].max(dim=-1)[0]
                    confs = (objectness * class_conf).numpy()
                elif predictions.shape[1] >= 5:
                    confs = predictions[:, 4].numpy()
                else:
                    confs = predictions.flatten().numpy()
                if self.conf_filter_min is not None:
                    confs = confs[confs > self.conf_filter_min]
                return confs
            elif predictions.dim() == 3:
                # (batch, num_det, features)
                # 最终置信度 = objectness * max(class_conf)
                if predictions.shape[2] >= 6:
                    objectness = predictions[..., 4]
                    class_conf = predictions[..., 5:].max(dim=-1)[0]
                    confs = (objectness * class_conf).flatten().numpy()
                else:
                    confs = predictions[..., 4].flatten().numpy()
                if self.conf_filter_min is not None:
                    confs = confs[confs > self.conf_filter_min]
                return confs
        
        return None
    
    def _otsu_threshold(self, scores: np.ndarray) -> float:
        """
        Otsu 方法计算最优阈值
        
        原理：最大化类间方差 σ_B² = w₀ * w₁ * (μ₀ - μ₁)²
        
        Args:
            scores: 置信度分数数组，范围 [0, 1]
            
        Returns:
            optimal_threshold: 最优阈值
        """
        # 过滤有效范围
        scores = scores[(scores >= 0) & (scores <= 1)]
        if len(scores) < 10:
            return self.init_threshold
        
        # 构建直方图
        hist, bin_edges = np.histogram(scores, bins=self.num_bins, range=(0, 1))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # 归一化为概率分布
        hist = hist.astype(np.float64)
        total = hist.sum()
        if total == 0:
            return self.init_threshold
        hist = hist / total
        
        # 计算累积和
        cumsum_prob = np.cumsum(hist)  # w₀(t)
        cumsum_mean = np.cumsum(hist * bin_centers)  # μ₀(t) * w₀(t)
        global_mean = cumsum_mean[-1]  # μ_T
        
        # 计算类间方差
        # σ_B²(t) = w₀(t) * w₁(t) * (μ₀(t) - μ₁(t))²
        #         = [μ_T * w₀(t) - μ(t)]² / [w₀(t) * (1 - w₀(t))]
        
        w0 = cumsum_prob[:-1]  # 排除最后一个点
        w1 = 1 - w0
        
        # 避免除零
        valid_mask = (w0 > 1e-10) & (w1 > 1e-10)
        
        if not valid_mask.any():
            return self.init_threshold
        
        mu0 = cumsum_mean[:-1] / (w0 + 1e-10)
        mu1 = (global_mean - cumsum_mean[:-1]) / (w1 + 1e-10)
        
        # 类间方差
        between_variance = w0 * w1 * (mu0 - mu1) ** 2
        between_variance[~valid_mask] = 0
        
        # 找到最大类间方差对应的阈值
        optimal_idx = np.argmax(between_variance)
        optimal_threshold = bin_centers[optimal_idx]
        
        return float(optimal_threshold)
    
    def get_confidence_histogram(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取最近一次的置信度直方图（用于可视化）"""
        if self.last_confidence_scores is None:
            return None, None
        hist, bin_edges = np.histogram(
            self.last_confidence_scores, 
            bins=self.num_bins, 
            range=(0, 1)
        )
        return hist, bin_edges
    
    def reset(self):
        """重置状态（新 epoch 开始时可选调用）"""
        self.batch_count = 0
        # 不重置 ema_threshold，保持连续性
        
    def state_dict(self) -> dict:
        """保存状态（用于 checkpoint）"""
        return {
            'current_threshold': self.current_threshold,
            'ema_threshold': self.ema_threshold,
            'batch_count': self.batch_count,
            'threshold_history': self.threshold_history
        }
    
    def load_state_dict(self, state: dict):
        """加载状态（从 checkpoint 恢复）"""
        self.current_threshold = state.get('current_threshold', self.init_threshold)
        self.ema_threshold = state.get('ema_threshold', self.init_threshold)
        self.batch_count = state.get('batch_count', 0)
        self.threshold_history = state.get('threshold_history', [])
    
    def __repr__(self):
        return (
            f"DAAT(threshold={self.current_threshold:.4f}, "
            f"ema={self.ema_threshold:.4f}, "
            f"bounds=[{self.lower_bound}, {self.upper_bound}], "
            f"batches={self.batch_count})"
        )


class DAATWithClassBalance(DAAT):
    """
    带类别平衡的 DAAT 变体
    
    为每个类别单独计算自适应阈值，解决类别不平衡问题。
    """
    
    def __init__(
        self,
        num_classes: int,
        init_threshold: float = 0.5,
        ema_momentum: float = 0.9,
        lower_bound: float = 0.25,
        upper_bound: float = 0.9,
        num_bins: int = 100,
        warmup_batches: int = 100
    ):
        super().__init__(
            init_threshold, ema_momentum, lower_bound, 
            upper_bound, num_bins, warmup_batches
        )
        self.num_classes = num_classes
        self.class_thresholds = [init_threshold] * num_classes
        self.class_ema_thresholds = [init_threshold] * num_classes
    
    def get_class_thresholds(
        self, 
        predictions: torch.Tensor,
        class_indices: Optional[torch.Tensor] = None
    ) -> List[float]:
        """
        获取每个类别的自适应阈值
        
        Args:
            predictions: 预测结果
            class_indices: 类别索引
            
        Returns:
            class_thresholds: 每个类别的阈值列表
        """
        # TODO: 实现类别级别的阈值计算
        # 当前返回全局阈值
        global_threshold = self.get_threshold(predictions)
        return [global_threshold] * self.num_classes


def create_daat(
    enabled: bool = True,
    init_threshold: float = 0.5,
    ema_momentum: float = 0.9,
    lower_bound: float = 0.25,
    upper_bound: float = 0.9,
    **kwargs
) -> Optional[DAAT]:
    """
    工厂函数：创建 DAAT 实例
    
    Args:
        enabled: 是否启用 DAAT
        其他参数同 DAAT
        
    Returns:
        DAAT 实例或 None
    """
    if not enabled:
        return None
    return DAAT(
        init_threshold=init_threshold,
        ema_momentum=ema_momentum,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        **kwargs
    )
