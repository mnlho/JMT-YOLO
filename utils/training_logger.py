"""
Training Logger for DAAT + ASA visualization
Records key metrics during training for paper figures
"""

import json
import os
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict


class TrainingLogger:
    """记录训练过程中的关键数据，用于论文可视化"""
    
    def __init__(self, save_dir, enabled=True):
        self.save_dir = Path(save_dir)
        self.enabled = enabled
        self.log_file = self.save_dir / 'training_stats.json'
        
        # 初始化数据存储
        self.data = {
            # DAAT 相关
            'epochs': [],                    # epoch 编号
            'adaptive_thresholds': [],      # 每个 epoch 的自适应阈值
            'confidence_distributions': [],  # 采样的置信度分布
            'pseudo_label_counts': [],       # 伪标签数量
            
            # ASA 相关 (如果使用)
            'dynamic_k_values': [],          # 动态 K 的分布
            'positive_sample_counts': [],    # 正样本匹配数量
        }
        
        # 临时存储（用于 batch 级别统计）
        self.batch_confidences = []
        self.batch_pseudo_counts = []
        self.batch_dynamic_k = []
        
    def log_confidence_scores(self, confidences):
        """记录一个 batch 的置信度分数"""
        if not self.enabled:
            return
        if isinstance(confidences, torch.Tensor):
            confidences = confidences.detach().cpu().numpy()
        self.batch_confidences.extend(confidences.flatten().tolist())
    
    def log_pseudo_label_count(self, count):
        """记录伪标签数量"""
        if not self.enabled:
            return
        self.batch_pseudo_counts.append(count)
    
    def log_adaptive_threshold(self, threshold):
        """记录当前使用的自适应阈值"""
        if not self.enabled:
            return
        # 这个在 epoch 结束时调用
        self.data['adaptive_thresholds'].append(float(threshold))
    
    def log_dynamic_k(self, k_values):
        """记录 ASA 的动态 K 值"""
        if not self.enabled:
            return
        if isinstance(k_values, torch.Tensor):
            k_values = k_values.detach().cpu().numpy()
        self.batch_dynamic_k.extend(k_values.flatten().tolist())
    
    def end_epoch(self, epoch):
        """epoch 结束时汇总 batch 级别的统计"""
        if not self.enabled:
            return
        
        # 记录 epoch 编号
        self.data['epochs'].append(epoch)
            
        # 汇总置信度分布（采样保存，避免数据过大）
        if self.batch_confidences:
            # 保存直方图数据而非原始数据
            hist, bin_edges = np.histogram(self.batch_confidences, bins=50, range=(0, 1))
            self.data['confidence_distributions'].append({
                'epoch': epoch,
                'histogram': hist.tolist(),
                'bin_edges': bin_edges.tolist(),
                'mean': float(np.mean(self.batch_confidences)),
                'std': float(np.std(self.batch_confidences)),
                'count': len(self.batch_confidences)
            })
        
        # 汇总伪标签数量
        if self.batch_pseudo_counts:
            self.data['pseudo_label_counts'].append({
                'epoch': epoch,
                'mean': float(np.mean(self.batch_pseudo_counts)),
                'std': float(np.std(self.batch_pseudo_counts)),
                'total': sum(self.batch_pseudo_counts)
            })
        
        # 汇总动态 K
        if self.batch_dynamic_k:
            self.data['dynamic_k_values'].append({
                'epoch': epoch,
                'mean': float(np.mean(self.batch_dynamic_k)),
                'std': float(np.std(self.batch_dynamic_k)),
                'histogram': np.histogram(self.batch_dynamic_k, bins=20)[0].tolist()
            })
        
        # 清空临时存储
        self.batch_confidences = []
        self.batch_pseudo_counts = []
        self.batch_dynamic_k = []
        
        # 每个 epoch 保存一次
        self.save()
    
    def save(self):
        """保存数据到 JSON 文件"""
        if not self.enabled:
            return
        with open(self.log_file, 'w') as f:
            json.dump(self.data, f, indent=2)
    
    def load(self, path=None):
        """加载已保存的数据"""
        path = path or self.log_file
        if os.path.exists(path):
            with open(path, 'r') as f:
                self.data = json.load(f)
        return self.data


def create_visualizations(log_path, output_dir=None):
    """
    从保存的训练日志创建可视化图表
    
    Usage:
        python -c "from utils.training_logger import create_visualizations; create_visualizations('runs/train/exp/training_stats.json')"
    """
    import matplotlib.pyplot as plt
    
    # 加载数据
    with open(log_path, 'r') as f:
        data = json.load(f)
    
    output_dir = Path(output_dir or Path(log_path).parent)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置绘图风格
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 16
    
    # ========== 图1: 自适应阈值变化曲线 ==========
    if data['adaptive_thresholds']:
        fig, ax = plt.subplots(figsize=(10, 6))
        epochs = range(1, len(data['adaptive_thresholds']) + 1)
        ax.plot(epochs, data['adaptive_thresholds'], 'b-', linewidth=2, marker='o', markersize=4)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Adaptive Threshold (τ)')
        ax.set_title('DAAT: Adaptive Threshold Evolution')
        ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Fixed threshold (baseline)')
        ax.legend()
        ax.set_ylim(0, 1)
        plt.tight_layout()
        plt.savefig(output_dir / 'daat_threshold_curve.png', dpi=300, bbox_inches='tight')
        plt.savefig(output_dir / 'daat_threshold_curve.pdf', bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: daat_threshold_curve.png/pdf")
    
    # ========== 图2: 置信度分布 + Otsu 分割点 ==========
    if data['confidence_distributions']:
        # 选择几个代表性的 epoch 展示
        n_dists = len(data['confidence_distributions'])
        selected_epochs = [0, n_dists//3, 2*n_dists//3, n_dists-1]
        selected_epochs = list(set([min(i, n_dists-1) for i in selected_epochs]))
        
        fig, axes = plt.subplots(1, len(selected_epochs), figsize=(4*len(selected_epochs), 4))
        if len(selected_epochs) == 1:
            axes = [axes]
        
        for ax, idx in zip(axes, selected_epochs):
            dist = data['confidence_distributions'][idx]
            bin_centers = [(dist['bin_edges'][i] + dist['bin_edges'][i+1])/2 
                          for i in range(len(dist['histogram']))]
            ax.bar(bin_centers, dist['histogram'], width=0.02, alpha=0.7, color='steelblue')
            
            # 标记阈值（如果有）
            if idx < len(data['adaptive_thresholds']):
                threshold = data['adaptive_thresholds'][idx]
                ax.axvline(x=threshold, color='red', linestyle='--', linewidth=2, 
                          label=f'τ={threshold:.3f}')
                ax.legend()
            
            ax.set_xlabel('Confidence Score')
            ax.set_ylabel('Count')
            ax.set_title(f'Epoch {dist["epoch"]}')
            ax.set_xlim(0, 1)
        
        plt.suptitle('Confidence Distribution with Otsu Threshold', y=1.02)
        plt.tight_layout()
        plt.savefig(output_dir / 'confidence_distribution.png', dpi=300, bbox_inches='tight')
        plt.savefig(output_dir / 'confidence_distribution.pdf', bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: confidence_distribution.png/pdf")
    
    # ========== 图3: 伪标签数量变化 ==========
    if data['pseudo_label_counts']:
        fig, ax = plt.subplots(figsize=(10, 6))
        epochs = [d['epoch'] for d in data['pseudo_label_counts']]
        means = [d['mean'] for d in data['pseudo_label_counts']]
        stds = [d['std'] for d in data['pseudo_label_counts']]
        
        ax.plot(epochs, means, 'g-', linewidth=2, marker='s', markersize=4, label='Mean')
        ax.fill_between(epochs, 
                        [m-s for m,s in zip(means, stds)], 
                        [m+s for m,s in zip(means, stds)], 
                        alpha=0.2, color='green')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Pseudo Labels per Image')
        ax.set_title('Pseudo Label Count Evolution')
        ax.legend()
        plt.tight_layout()
        plt.savefig(output_dir / 'pseudo_label_count.png', dpi=300, bbox_inches='tight')
        plt.savefig(output_dir / 'pseudo_label_count.pdf', bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: pseudo_label_count.png/pdf")
    
    # ========== 图4: 动态 K 分布 (ASA) ==========
    if data['dynamic_k_values']:
        fig, ax = plt.subplots(figsize=(10, 6))
        epochs = [d['epoch'] for d in data['dynamic_k_values']]
        means = [d['mean'] for d in data['dynamic_k_values']]
        stds = [d['std'] for d in data['dynamic_k_values']]
        
        ax.plot(epochs, means, 'purple', linewidth=2, marker='d', markersize=4, label='Mean K')
        ax.fill_between(epochs, 
                        [m-s for m,s in zip(means, stds)], 
                        [m+s for m,s in zip(means, stds)], 
                        alpha=0.2, color='purple')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Dynamic K')
        ax.set_title('ASA: Dynamic K Value Evolution')
        ax.legend()
        plt.tight_layout()
        plt.savefig(output_dir / 'asa_dynamic_k.png', dpi=300, bbox_inches='tight')
        plt.savefig(output_dir / 'asa_dynamic_k.pdf', bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: asa_dynamic_k.png/pdf")
    
    print(f"\n✅ All visualizations saved to: {output_dir}")


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        create_visualizations(sys.argv[1])
    else:
        print("Usage: python training_logger.py <path_to_training_stats.json>")
