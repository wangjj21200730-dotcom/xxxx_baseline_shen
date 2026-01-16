"""
GRAM-C: Collaborative-Enhanced GRAM

核心改进：将协同过滤信号（来自 LightGCN）作为"协同上下文前缀"注入到 T5 Encoder，
而非直接与语义向量相加。

关键 Tricks:
1. Prefix Scaling - 初始缩放因子防止早期训练崩溃
2. Recent-aware Prefix - 使用最近 K 个物品的 GCN embedding，解决长短期兴趣冲突
3. Prefix Dropout - 训练时随机丢弃前缀，防止过度依赖协同信号
"""

import torch
import torch.nn as nn
import logging
import os

from .gram import GRAM, EncoderWrapper, apply_checkpoint_wrapper
from .collaborative_adapter import CollaborativeAdapter
from .gram_t5_outputs import BaseModelOutput


class GRAM_C(GRAM):
    """
    GRAM-C: 带协同上下文前缀的 GRAM 模型
    
    新增属性:
        - gcn_item_embedding: nn.Embedding, 预训练的 Item GCN 向量（冻结）
        - collaborative_adapter: CollaborativeAdapter
        - use_collaborative_prefix: bool
        - use_sequential_prefix: bool (新增：是否使用序列化前缀)
        - prefix_dropout_prob: float
        - recent_k: int
    """
    
    def __init__(self, config):
        # 先调用父类初始化
        super().__init__(config)
        
        # GRAM-C 配置
        self.use_collaborative_prefix = getattr(config, 'use_collaborative_prefix', True)
        self.use_sequential_prefix = getattr(config, 'use_sequential_prefix', False)  # 新增
        self.prefix_dropout_prob = getattr(config, 'prefix_dropout_prob', 0.2)
        self.recent_k = getattr(config, 'recent_k', 5)
        self.gcn_dim = getattr(config, 'gcn_dim', 64)
        self.prefix_init_scale = getattr(config, 'prefix_init_scale', 0.1)
        self.adapter_dropout = getattr(config, 'adapter_dropout', 0.1)
        
        if self.use_collaborative_prefix:
            # GCN Item Embedding (从预训练文件加载，训练时冻结)
            num_items = getattr(config, 'num_items', 10000) + 1  # +1 for padding/unknown
            self.gcn_item_embedding = nn.Embedding(num_items, self.gcn_dim)
            self.gcn_item_embedding.weight.requires_grad = False  # 冻结
            
            # 初始化第 0 行为零向量（用于 padding/unknown）
            with torch.no_grad():
                self.gcn_item_embedding.weight[0].zero_()
            
            # 协同适配器
            self.collaborative_adapter = CollaborativeAdapter(
                gcn_dim=self.gcn_dim,
                llm_dim=config.d_model,
                dropout_rate=self.adapter_dropout,
                init_scale=self.prefix_init_scale
            )
            
            logging.info(f"GRAM-C initialized with collaborative prefix:")
            logging.info(f"  - GCN dim: {self.gcn_dim}")
            logging.info(f"  - LLM dim: {config.d_model}")
            logging.info(f"  - Recent K: {self.recent_k}")
            logging.info(f"  - Sequential prefix: {self.use_sequential_prefix}")
            if self.use_sequential_prefix:
                logging.info(f"    -> Using {self.recent_k} sequential tokens (no pooling)")
            else:
                logging.info(f"    -> Using mean pooling (single token)")
            logging.info(f"  - Prefix init scale: {self.prefix_init_scale}")
            logging.info(f"  - Prefix dropout prob: {self.prefix_dropout_prob}")
        else:
            self.gcn_item_embedding = None
            self.collaborative_adapter = None
            logging.info("GRAM-C initialized without collaborative prefix (fallback to GRAM)")
        
        # 重新包装 encoder 以支持协同前缀
        self.wrap_encoder_c()
    
    def wrap_encoder_c(self, use_checkpoint=False):
        """
        重新包装 T5 encoder 以支持协同前缀注入
        """
        # 先解包（如果已经包装过）
        if hasattr(self.encoder, 'encoder'):
            self.encoder = self.encoder.encoder
            # 解包 checkpoint wrapper
            block = []
            for mod in self.encoder.block:
                if hasattr(mod, 'module'):
                    block.append(mod.module)
                else:
                    block.append(mod)
            block = nn.ModuleList(block)
            self.encoder.block = block
        
        # 使用新的 EncoderWrapperC
        self.encoder = EncoderWrapperC(
            encoder=self.encoder,
            config=self.config,
            use_checkpoint=use_checkpoint,
            position_embedding=self.position_embedding,
            collaborative_adapter=self.collaborative_adapter,
            gcn_item_embedding=self.gcn_item_embedding,
            use_collaborative_prefix=self.use_collaborative_prefix,
            use_sequential_prefix=self.use_sequential_prefix,  # 新增
            prefix_dropout_prob=self.prefix_dropout_prob,
        )

    def wrap_encoder(self, use_checkpoint=False):
        # During GRAM.__init__, wrap_encoder() is called before GRAM_C finishes
        # initializing collaborative modules. Fall back to the parent wrapper
        # in that early stage, and use EncoderWrapperC afterwards.
        if not hasattr(self, "collaborative_adapter") or not hasattr(
            self, "gcn_item_embedding"
        ):
            return GRAM.wrap_encoder(self, use_checkpoint=use_checkpoint)
        return self.wrap_encoder_c(use_checkpoint=use_checkpoint)

    def load_t5(self, state_dict):
        # Use parent loading logic, then re-wrap encoder with EncoderWrapperC.
        super().load_t5(state_dict)
        if self.use_collaborative_prefix:
            self.wrap_encoder_c()
    
    def load_gcn_embeddings(self, gcn_emb_path: str):
        """
        从文件加载预训练的 GCN Item Embeddings
        
        Args:
            gcn_emb_path: GCN embedding 文件路径 (.pt 或 .npy)
        """
        if not self.use_collaborative_prefix:
            logging.warning("Collaborative prefix is disabled, skipping GCN embedding loading")
            return
        
        if not os.path.exists(gcn_emb_path):
            raise FileNotFoundError(f"GCN embedding file not found: {gcn_emb_path}")
        
        if gcn_emb_path.endswith('.pt'):
            gcn_embeddings = torch.load(gcn_emb_path, map_location='cpu')
        elif gcn_emb_path.endswith('.npy'):
            import numpy as np
            gcn_embeddings = torch.from_numpy(np.load(gcn_emb_path))
        else:
            raise ValueError(f"Unsupported GCN embedding file format: {gcn_emb_path}")
        
        # 验证维度
        if gcn_embeddings.shape[1] != self.gcn_dim:
            raise ValueError(
                f"GCN embedding dimension mismatch: "
                f"expected {self.gcn_dim}, got {gcn_embeddings.shape[1]}"
            )
        
        # 验证第 0 行是否为零向量
        if not torch.allclose(gcn_embeddings[0], torch.zeros(self.gcn_dim)):
            logging.warning("GCN embedding row 0 is not zero vector, forcing it to zero")
            gcn_embeddings[0] = 0
        
        # 加载到 embedding 层
        with torch.no_grad():
            num_items = min(gcn_embeddings.shape[0], self.gcn_item_embedding.weight.shape[0])
            self.gcn_item_embedding.weight[:num_items] = gcn_embeddings[:num_items]
        
        logging.info(f"Loaded GCN embeddings from {gcn_emb_path}")
        logging.info(f"  - Shape: {gcn_embeddings.shape}")
        logging.info(f"  - Loaded items: {num_items}")
    
    def forward(self, input_ids=None, attention_mask=None, recent_item_ids=None, **kwargs):
        """
        前向传播
        
        Args:
            input_ids: [Batch, N, L] 或 [Batch, N*L]
            attention_mask: [Batch, N, L] 或 [Batch, N*L]
            recent_item_ids: [Batch, K] 最近 K 个物品的 GCN 索引（GRAM-C 新增）
            **kwargs: 其他参数传递给父类
        """
        # 如果 encoder_outputs 已经存在（generation 的 decoder 阶段），直接使用
        if kwargs.get('encoder_outputs') is not None:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **kwargs
            )
        
        # 设置 recent_item_ids 到 encoder
        if hasattr(self.encoder, 'set_recent_item_ids'):
            self.encoder.set_recent_item_ids(recent_item_ids)

        input_ids2 = input_ids
        attention_mask2 = attention_mask
        if input_ids2 is not None and input_ids2.dim() == 3:
            self.encoder.n_passages = input_ids2.size(1)
            input_ids2 = input_ids2.view(input_ids2.size(0), -1)
        if attention_mask2 is not None and attention_mask2.dim() == 3:
            attention_mask2 = attention_mask2.view(attention_mask2.size(0), -1)

        last_hidden_states = self.encoder(
            input_ids=input_ids2,
            attention_mask=attention_mask2,
            return_dict=True,
        )[0]

        encoder_attention_mask = None
        if hasattr(self.encoder, 'get_last_encoder_attention_mask'):
            encoder_attention_mask = self.encoder.get_last_encoder_attention_mask()
        if encoder_attention_mask is None:
            encoder_attention_mask = attention_mask2

        encoder_outputs = BaseModelOutput(
            last_hidden_state=last_hidden_states
        )

        input_ids_for_model = input_ids2
        if (
            input_ids_for_model is not None
            and encoder_attention_mask is not None
            and input_ids_for_model.size(1) != encoder_attention_mask.size(1)
        ):
            diff = encoder_attention_mask.size(1) - input_ids_for_model.size(1)
            if diff > 0:
                pad = torch.zeros(
                    (input_ids_for_model.size(0), diff),
                    dtype=input_ids_for_model.dtype,
                    device=input_ids_for_model.device,
                )
                input_ids_for_model = torch.cat([pad, input_ids_for_model], dim=1)

        return super().forward(
            input_ids=input_ids_for_model,
            attention_mask=encoder_attention_mask,
            encoder_outputs=encoder_outputs,
            **kwargs,
        )
    
    def generate(self, input_ids, attention_mask, max_length, recent_item_ids=None, **kwargs):
        """
        生成方法
        
        Args:
            input_ids: [Batch, N, L]
            attention_mask: [Batch, N, L]
            max_length: 最大生成长度
            recent_item_ids: [Batch, K] 最近 K 个物品的 GCN 索引
            **kwargs: 其他参数
        """
        # 设置 recent_item_ids 到 encoder
        if hasattr(self.encoder, 'set_recent_item_ids'):
            self.encoder.set_recent_item_ids(recent_item_ids)

        self.encoder.n_passages = input_ids.size(1)
        input_ids2 = input_ids.view(input_ids.size(0), -1)
        attention_mask2 = attention_mask.view(attention_mask.size(0), -1) if attention_mask is not None else None

        last_hidden_states = self.encoder(
            input_ids=input_ids2,
            attention_mask=attention_mask2,
            return_dict=True,
        )[0]

        encoder_attention_mask = None
        if hasattr(self.encoder, 'get_last_encoder_attention_mask'):
            encoder_attention_mask = self.encoder.get_last_encoder_attention_mask()
        if encoder_attention_mask is None:
            encoder_attention_mask = attention_mask2
        
        encoder_outputs = BaseModelOutput(
            last_hidden_state=last_hidden_states
        )

        input_ids_for_generate = input_ids2
        if (
            input_ids_for_generate is not None
            and encoder_attention_mask is not None
            and input_ids_for_generate.size(1) != encoder_attention_mask.size(1)
        ):
            diff = encoder_attention_mask.size(1) - input_ids_for_generate.size(1)
            if diff > 0:
                pad = torch.zeros(
                    (input_ids_for_generate.size(0), diff),
                    dtype=input_ids_for_generate.dtype,
                    device=input_ids_for_generate.device,
                )
                input_ids_for_generate = torch.cat([pad, input_ids_for_generate], dim=1)

        # 调用 T5 的 generate
        from transformers import T5ForConditionalGeneration
        outputs = T5ForConditionalGeneration.generate(
            self,
            input_ids=input_ids_for_generate,
            attention_mask=encoder_attention_mask,
            encoder_outputs=encoder_outputs,
            max_length=max_length,
            **kwargs
        )

        if kwargs.get("output_hidden_states"):
            encoder_outputs["last_hidden_state"] = encoder_outputs.last_hidden_state[0:1, :, :]
            outputs["encoder_outputs"] = encoder_outputs

        return outputs


class EncoderWrapperC(EncoderWrapper):
    """
    支持协同前缀注入的 EncoderWrapper
    
    关键设计：Pre-Encoder Prefix Injection
    ========================================
    Collaborative prefix 必须在 encoder.forward() 之前注入到 embedding 层，
    作为 encoder self-attention 的一部分存在。
    
    正确的流程：
    1. 获取 input embedding: inputs_embeds = embed_tokens(input_ids)
    2. 构建 collaborative prefix: prefix_emb = adapter(gcn_pooled)  [B, P, D]
       - P=1 (均值池化模式) 或 P=K (序列化模式)
    3. 在 embedding 层拼接: inputs_embeds = cat([prefix_emb, inputs_embeds], dim=1)
    4. 扩展 attention_mask: attention_mask = cat([prefix_mask, attention_mask], dim=1)
    5. 送入 encoder: encoder(inputs_embeds=..., attention_mask=...)
    
    这样 prefix 才能参与 encoder 的 self-attention 计算。
    """
    
    def __init__(
        self, 
        encoder, 
        config=None, 
        use_checkpoint=False, 
        position_embedding=None,
        collaborative_adapter=None,
        gcn_item_embedding=None,
        use_collaborative_prefix=True,
        use_sequential_prefix=False,  # 新增：是否使用序列化前缀
        prefix_dropout_prob=0.2,
    ):
        super().__init__(encoder, config, use_checkpoint, position_embedding)
        
        self.collaborative_adapter = collaborative_adapter
        self.gcn_item_embedding = gcn_item_embedding
        self.use_collaborative_prefix = use_collaborative_prefix
        self.use_sequential_prefix = use_sequential_prefix  # 新增
        self.prefix_dropout_prob = prefix_dropout_prob
        
        # 用于存储当前 batch 的 recent_item_ids
        self._recent_item_ids = None
        self._last_encoder_attention_mask = None
    
    def set_recent_item_ids(self, recent_item_ids):
        """设置当前 batch 的 recent_item_ids"""
        self._recent_item_ids = recent_item_ids

    def get_last_encoder_attention_mask(self):
        return self._last_encoder_attention_mask
    
    def _compute_collaborative_prefix(self, recent_item_ids, device):
        """
        计算协同前缀 embedding
        
        Args:
            recent_item_ids: [Batch, K] 最近 K 个物品的 GCN 索引
            device: 目标设备
            
        Returns:
            如果 use_sequential_prefix=False:
                [Batch, 1, LLM_Dim] 均值池化后的单 token
            如果 use_sequential_prefix=True:
                [Batch, K, LLM_Dim] K 个独立的 soft tokens
        """
        if recent_item_ids is None or self.gcn_item_embedding is None:
            return None
        
        # 查找 GCN embeddings: [Batch, K, GCN_Dim]
        recent_item_ids = recent_item_ids.to(device)
        gcn_embs = self.gcn_item_embedding(recent_item_ids)
        
        if self.use_sequential_prefix:
            # 序列化模式：每个 token 独立通过 adapter
            # gcn_embs: [Batch, K, GCN_Dim]
            # CollaborativeAdapter 支持 3D 输入，对最后一维进行变换
            prefix_emb = self.collaborative_adapter(gcn_embs)
            # prefix_emb: [Batch, K, LLM_Dim]
            
            # 整段 Prefix Dropout（K 个 tokens 一起置零）
            if self.training and self.prefix_dropout_prob > 0:
                # 生成 [Batch, 1, 1] 的 dropout mask，广播到所有 K 个 tokens
                dropout_mask = (
                    torch.rand(prefix_emb.shape[0], 1, 1, device=device) > self.prefix_dropout_prob
                ).float()
                prefix_emb = prefix_emb * dropout_mask
        else:
            # 原有模式：均值池化
            # 注意：索引为 0 的是 padding，不应该参与 pooling
            mask = (recent_item_ids != 0).float().unsqueeze(-1)  # [Batch, K, 1]
            valid_counts = mask.sum(dim=1).clamp(min=1)  # [Batch, 1]
            pooled_emb = (gcn_embs * mask).sum(dim=1) / valid_counts  # [Batch, GCN_Dim]
            
            # 通过 Adapter 映射到 LLM 空间: [Batch, LLM_Dim]
            prefix_emb = self.collaborative_adapter(pooled_emb)
            
            # Prefix Dropout (仅在训练时)
            if self.training and self.prefix_dropout_prob > 0:
                dropout_mask = (
                    torch.rand(prefix_emb.shape[0], 1, device=device) > self.prefix_dropout_prob
                ).float()
                prefix_emb = prefix_emb * dropout_mask
            
            # 添加序列维度: [Batch, 1, LLM_Dim]
            prefix_emb = prefix_emb.unsqueeze(1)
        
        return prefix_emb
    
    def forward(
        self, input_ids=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        """
        前向传播，支持 Pre-Encoder Prefix Injection
        
        关键：prefix 在 encoder.forward() 之前注入到 embedding 层
        
        支持两种模式：
        - 均值池化模式 (use_sequential_prefix=False): prefix 形状为 [B*N, 1, D]
        - 序列化模式 (use_sequential_prefix=True): prefix 形状为 [B*N, K, D]
        """
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        
        # Step 1: 计算 batch size 和 passage 信息
        if input_ids is not None:
            bsz, total_length = input_ids.shape
            passage_length = total_length // self.n_passages
        else:
            bsz, total_length, _ = inputs_embeds.shape
            passage_length = total_length // self.n_passages
        
        # Step 2: 计算 collaborative prefix（在 encoder 之前！）
        # prefix_emb: [B, P, D] 其中 P=1 (均值池化) 或 P=K (序列化)
        prefix_emb = None
        if self.use_collaborative_prefix and self._recent_item_ids is not None:
            prefix_emb = self._compute_collaborative_prefix(self._recent_item_ids, device)
        
        # Step 3: 获取 input embeddings
        if input_ids is not None:
            # 重塑 input_ids: [Batch * N_passages, Passage_Length]
            input_ids_reshaped = input_ids.view(bsz * self.n_passages, passage_length)
            attention_mask_reshaped = attention_mask.view(bsz * self.n_passages, passage_length)
            
            # 获取 word embeddings（关键：使用 embed_tokens 而不是直接调用 encoder）
            inputs_embeds_reshaped = self.encoder.embed_tokens(input_ids_reshaped)
            # [Batch * N_passages, Passage_Length, D]
            
        else:
            inputs_embeds_reshaped = inputs_embeds.view(bsz * self.n_passages, passage_length, -1)
            attention_mask_reshaped = attention_mask.view(bsz * self.n_passages, passage_length)
        
        # Step 4: Pre-Encoder Prefix Injection
        # Fix: keep FiD-style independent passage encoding to avoid OOM.
        if prefix_emb is not None:
            # prefix_emb: [B, P, D] -> [B*N, P, D]
            # P = 1 (均值池化) 或 P = K (序列化)
            prefix_length = prefix_emb.shape[1]
            prefix_emb_expanded = prefix_emb.repeat_interleave(self.n_passages, dim=0)

            inputs_embeds_with_prefix = torch.cat(
                [prefix_emb_expanded, inputs_embeds_reshaped], dim=1
            )  # [B*N, P+L, D]

            prefix_mask = torch.ones(
                (bsz * self.n_passages, prefix_length),
                dtype=attention_mask_reshaped.dtype,
                device=device,
            )
            attention_mask_with_prefix = torch.cat(
                [prefix_mask, attention_mask_reshaped], dim=1
            )  # [B*N, P+L]

            self._last_encoder_attention_mask = attention_mask_with_prefix.view(
                bsz, self.n_passages * (passage_length + prefix_length)
            )

            outputs = self.encoder(
                inputs_embeds=inputs_embeds_with_prefix,
                attention_mask=attention_mask_with_prefix,
                **kwargs,
            )
            last_hidden_states = outputs[0]  # [B*N, P+L, D]

            if self.position_embedding is not None:
                position_ids = torch.arange(self.n_passages, device=device).expand(
                    bsz, self.n_passages
                )
                position_embeddings = self.position_embedding(position_ids)
                position_embeddings = position_embeddings.view(
                    bsz * self.n_passages, 1, -1
                )
                last_hidden_states = last_hidden_states + position_embeddings

            last_hidden_states = last_hidden_states.view(
                bsz, self.n_passages * (passage_length + prefix_length), -1
            )

        else:
            outputs = self.encoder(
                inputs_embeds=inputs_embeds_reshaped,
                attention_mask=attention_mask_reshaped,
                **kwargs,
            )

            last_hidden_states = outputs[0]

            if self.position_embedding is not None:
                position_ids = torch.arange(self.n_passages, device=device).expand(
                    bsz, self.n_passages
                )
                position_embeddings = self.position_embedding(position_ids)
                position_embeddings = position_embeddings.view(
                    bsz * self.n_passages, 1, -1
                )
                last_hidden_states = last_hidden_states + position_embeddings

            last_hidden_states = last_hidden_states.view(
                bsz, self.n_passages * passage_length, -1
            )

            self._last_encoder_attention_mask = attention_mask_reshaped.view(
                bsz, self.n_passages * passage_length
            )
        
        outputs = (last_hidden_states,) + outputs[1:]
        return outputs
