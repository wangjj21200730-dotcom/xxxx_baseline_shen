"""
CollaborativeAdapter: 协同适配器模块

将 GCN embedding 对齐到 T5 语义空间，用于 GRAM-C 的协同上下文前缀注入。

核心设计原则：
1. LayerNorm 归一化解决尺度问题
2. 两层 MLP 投影实现空间对齐
3. init_scale 缩放防止早期训练 attention collapse
4. Dropout 防止过拟合
"""

import torch
import torch.nn as nn


class CollaborativeAdapter(nn.Module):
    """
    协同适配器：将 GCN embedding 对齐到 T5 语义空间
    
    Args:
        gcn_dim: GCN embedding 维度 (默认 64)
        llm_dim: T5 hidden dimension (默认 512 for t5-small)
        dropout_rate: Dropout 率 (默认 0.1)
        init_scale: 输出缩放因子 (默认 0.1)，防止早期训练 attention collapse
    """
    
    def __init__(
        self, 
        gcn_dim: int, 
        llm_dim: int, 
        dropout_rate: float = 0.1, 
        init_scale: float = 0.1
    ):
        super().__init__()
        self.gcn_dim = gcn_dim
        self.llm_dim = llm_dim
        self.init_scale = init_scale
        
        # 1. 输入归一化：将 GCN 向量拉回标准分布，解决尺度问题
        self.input_norm = nn.LayerNorm(gcn_dim)
        
        # 2. 两层 MLP 投影：对齐两个空间 (Space Alignment)
        self.projector = nn.Sequential(
            nn.Linear(gcn_dim, llm_dim),
            nn.GELU(),  # 激活函数增加非线性
            nn.Linear(llm_dim, llm_dim)  # 二次映射增加拟合能力
        )
        
        # 3. 输出归一化与 Dropout：防止过拟合和数值爆炸
        self.output_norm = nn.LayerNorm(llm_dim)
        self.dropout = nn.Dropout(dropout_rate)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """使用标准初始化"""
        for module in self.projector:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, gcn_emb: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            gcn_emb: [Batch, GCN_Dim] 或 [Batch, K, GCN_Dim] 的 GCN embedding
            
        Returns:
            [Batch, LLM_Dim] 或 [Batch, K, LLM_Dim] 对齐后的向量，已应用 scaling
        """
        # 输入归一化
        x = self.input_norm(gcn_emb)
        
        # MLP 投影
        x = self.projector(x)
        
        # 输出归一化
        x = self.output_norm(x)
        
        # Dropout
        x = self.dropout(x)
        
        # 应用缩放因子（关键 Trick：防止早期训练 attention collapse）
        return x * self.init_scale
    
    def extra_repr(self) -> str:
        return f'gcn_dim={self.gcn_dim}, llm_dim={self.llm_dim}, init_scale={self.init_scale}'
