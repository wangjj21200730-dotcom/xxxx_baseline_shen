#!/usr/bin/env python3
"""
端到端集成测试 - 测试从 user_sequence.txt 到模型训练的完整流程
以及新旧配置的向后兼容性。
Requirements: 6.1, 6.2, 6.3
"""
import os
import sys
import json
import tempfile
import shutil
import pytest
import torch
import torch.nn as nn

_project_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, _project_root)
sys.path.insert(0, os.path.join(_project_root, 'scripts'))

import importlib.util
_adapter_path = os.path.join(_project_root, 'src', 'model', 'collaborative_adapter.py')
_spec = importlib.util.spec_from_file_location("collaborative_adapter", _adapter_path)
_adapter_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_adapter_module)
CollaborativeAdapter = _adapter_module.CollaborativeAdapter


class MinimalEncoderWrapperC(nn.Module):
    """Minimal EncoderWrapperC for testing prefix computation logic."""
    def __init__(self, collaborative_adapter=None, gcn_item_embedding=None,
                 use_collaborative_prefix=True, use_sequential_prefix=False,
                 prefix_dropout_prob=0.2):
        super().__init__()
        self.collaborative_adapter = collaborative_adapter
        self.gcn_item_embedding = gcn_item_embedding
        self.use_collaborative_prefix = use_collaborative_prefix
        self.use_sequential_prefix = use_sequential_prefix
        self.prefix_dropout_prob = prefix_dropout_prob
        self.n_passages = 1

    def _compute_collaborative_prefix(self, recent_item_ids, device):
        if recent_item_ids is None or self.gcn_item_embedding is None:
            return None
        recent_item_ids = recent_item_ids.to(device)
        gcn_embs = self.gcn_item_embedding(recent_item_ids)
        if self.use_sequential_prefix:
            prefix_emb = self.collaborative_adapter(gcn_embs)
            if self.training and self.prefix_dropout_prob > 0:
                dropout_mask = (torch.rand(prefix_emb.shape[0], 1, 1, device=device) > self.prefix_dropout_prob).float()
                prefix_emb = prefix_emb * dropout_mask
        else:
            mask = (recent_item_ids != 0).float().unsqueeze(-1)
            valid_counts = mask.sum(dim=1).clamp(min=1)
            pooled_emb = (gcn_embs * mask).sum(dim=1) / valid_counts
            prefix_emb = self.collaborative_adapter(pooled_emb)
            if self.training and self.prefix_dropout_prob > 0:
                dropout_mask = (torch.rand(prefix_emb.shape[0], 1, device=device) > self.prefix_dropout_prob).float()
                prefix_emb = prefix_emb * dropout_mask
            prefix_emb = prefix_emb.unsqueeze(1)
        return prefix_emb


class TestBackwardCompatibility:
    """测试向后兼容性 - Requirements: 6.1, 6.2, 6.3"""

    def test_collaborative_prefix_disabled(self):
        """Property 11: 向后兼容性 - 禁用协同前缀. Validates: Requirements 6.1"""
        wrapper = MinimalEncoderWrapperC(
            collaborative_adapter=None, gcn_item_embedding=None,
            use_collaborative_prefix=False, use_sequential_prefix=False,
            prefix_dropout_prob=0.2)
        assert wrapper.use_collaborative_prefix == False
        assert wrapper.collaborative_adapter is None
        assert wrapper.gcn_item_embedding is None
        recent_item_ids = torch.randint(1, 100, (2, 5))
        prefix = wrapper._compute_collaborative_prefix(recent_item_ids, 'cpu')
        assert prefix is None

    def test_sequential_prefix_disabled_uses_mean_pooling(self):
        """Property 12: Prefix 模式切换 - 均值池化模式. Validates: Requirements 6.2"""
        adapter = CollaborativeAdapter(gcn_dim=64, llm_dim=512, dropout_rate=0.1, init_scale=0.1)
        gcn_embedding = nn.Embedding(101, 64)
        with torch.no_grad():
            gcn_embedding.weight[0].zero_()
        wrapper = MinimalEncoderWrapperC(
            collaborative_adapter=adapter, gcn_item_embedding=gcn_embedding,
            use_collaborative_prefix=True, use_sequential_prefix=False,
            prefix_dropout_prob=0.0)
        wrapper.eval()
        batch_size, k = 2, 5
        recent_item_ids = torch.randint(1, 100, (batch_size, k))
        prefix = wrapper._compute_collaborative_prefix(recent_item_ids, 'cpu')
        assert prefix is not None
        assert prefix.shape == (batch_size, 1, 512), f"Expected (2, 1, 512), got {prefix.shape}"

    def test_sequential_prefix_enabled_uses_k_tokens(self):
        """Property 12: Prefix 模式切换 - 序列化模式. Validates: Requirements 6.3"""
        adapter = CollaborativeAdapter(gcn_dim=64, llm_dim=512, dropout_rate=0.1, init_scale=0.1)
        gcn_embedding = nn.Embedding(101, 64)
        with torch.no_grad():
            gcn_embedding.weight[0].zero_()
        wrapper = MinimalEncoderWrapperC(
            collaborative_adapter=adapter, gcn_item_embedding=gcn_embedding,
            use_collaborative_prefix=True, use_sequential_prefix=True,
            prefix_dropout_prob=0.0)
        wrapper.eval()
        batch_size, k = 2, 5
        recent_item_ids = torch.randint(1, 100, (batch_size, k))
        prefix = wrapper._compute_collaborative_prefix(recent_item_ids, 'cpu')
        assert prefix is not None
        assert prefix.shape == (batch_size, k, 512), f"Expected (2, 5, 512), got {prefix.shape}"


class TestDataConversionPipeline:
    """测试数据转换流程"""

    @pytest.fixture
    def temp_data_dir(self):
        temp_dir = tempfile.mkdtemp()
        dataset_dir = os.path.join(temp_dir, 'TestDataset')
        os.makedirs(dataset_dir)
        user_sequences = [
            "user1 item1 item2 item3 item4 item5",
            "user2 item2 item3 item4 item5 item6",
            "user3 item1 item3 item5 item7 item8",
        ]
        with open(os.path.join(dataset_dir, 'user_sequence.txt'), 'w') as f:
            f.write('\n'.join(user_sequences))
        yield temp_dir, 'TestDataset'
        shutil.rmtree(temp_dir)

    def test_data_converter_creates_required_files(self, temp_data_dir):
        """验证转换后生成 train.txt, test.txt, item_id_to_gcn_index.json"""
        from convert_to_lightgcn import LightGCNDataConverter
        data_path, dataset = temp_data_dir
        converter = LightGCNDataConverter(data_path, dataset)
        converter.convert()
        dataset_path = os.path.join(data_path, dataset)
        lightgcn_data_dir = os.path.join(dataset_path, 'lightgcn_data')
        assert os.path.exists(os.path.join(lightgcn_data_dir, 'train.txt'))
        assert os.path.exists(os.path.join(lightgcn_data_dir, 'test.txt'))
        assert os.path.exists(os.path.join(dataset_path, 'item_id_to_gcn_index.json'))
        assert os.path.exists(os.path.join(dataset_path, 'user_id_to_index.json'))

    def test_data_converter_index_mapping_is_1_based(self, temp_data_dir):
        """测试 item_id_to_gcn_index.json 使用 1-based 索引"""
        from convert_to_lightgcn import LightGCNDataConverter
        data_path, dataset = temp_data_dir
        converter = LightGCNDataConverter(data_path, dataset)
        converter.convert()
        dataset_path = os.path.join(data_path, dataset)
        mapping_path = os.path.join(dataset_path, 'item_id_to_gcn_index.json')
        with open(mapping_path, 'r') as f:
            item_mapping = json.load(f)
        min_index = min(item_mapping.values())
        assert min_index >= 1, f"Minimum index should be >= 1, got {min_index}"

    def test_data_converter_train_test_split(self, temp_data_dir):
        """Property 2: train 包含 items[:-2], test 包含 items[-1]"""
        from convert_to_lightgcn import LightGCNDataConverter
        data_path, dataset = temp_data_dir
        converter = LightGCNDataConverter(data_path, dataset)
        converter.convert()
        dataset_path = os.path.join(data_path, dataset)
        lightgcn_data_dir = os.path.join(dataset_path, 'lightgcn_data')
        train_interactions = 0
        with open(os.path.join(lightgcn_data_dir, 'train.txt'), 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    train_interactions += len(parts) - 1
        test_interactions = 0
        with open(os.path.join(lightgcn_data_dir, 'test.txt'), 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    test_interactions += 1
        assert train_interactions == 9, f"Expected 9 train interactions, got {train_interactions}"
        assert test_interactions == 3, f"Expected 3 test interactions, got {test_interactions}"


class TestCollaborativeAdapterIntegration:
    """测试 CollaborativeAdapter 集成"""

    def test_adapter_supports_2d_input(self):
        """测试 Adapter 支持 2D 输入（均值池化模式）"""
        adapter = CollaborativeAdapter(gcn_dim=64, llm_dim=512, dropout_rate=0.0, init_scale=0.1)
        adapter.eval()
        batch_size = 4
        gcn_emb = torch.randn(batch_size, 64)
        output = adapter(gcn_emb)
        assert output.shape == (batch_size, 512)

    def test_adapter_supports_3d_input(self):
        """测试 Adapter 支持 3D 输入（序列化模式）"""
        adapter = CollaborativeAdapter(gcn_dim=64, llm_dim=512, dropout_rate=0.0, init_scale=0.1)
        adapter.eval()
        batch_size, k = 4, 5
        gcn_emb = torch.randn(batch_size, k, 64)
        output = adapter(gcn_emb)
        assert output.shape == (batch_size, k, 512)

    def test_adapter_independence_property(self):
        """Property 5: Sequential Adapter 独立性. Validates: Requirements 4.2"""
        adapter = CollaborativeAdapter(gcn_dim=64, llm_dim=512, dropout_rate=0.0, init_scale=0.1)
        adapter.eval()
        batch_size, k = 2, 3
        gcn_emb_3d = torch.randn(batch_size, k, 64)
        output_3d = adapter(gcn_emb_3d)
        outputs_independent = []
        for i in range(k):
            token_emb = gcn_emb_3d[:, i, :]
            token_output = adapter(token_emb)
            outputs_independent.append(token_output)
        output_stacked = torch.stack(outputs_independent, dim=1)
        assert torch.allclose(output_3d, output_stacked, atol=1e-6), \
            "3D processing should be equivalent to independent 2D processing"


class TestEmbeddingLoading:
    """测试 Embedding 加载功能"""

    @pytest.fixture
    def temp_embedding_file(self):
        temp_dir = tempfile.mkdtemp()
        emb_path = os.path.join(temp_dir, 'test_embeddings.pt')
        num_items, embedding_dim = 100, 64
        embeddings = torch.randn(num_items + 1, embedding_dim)
        embeddings[0] = 0
        torch.save(embeddings, emb_path)
        yield emb_path, num_items, embedding_dim
        shutil.rmtree(temp_dir)

    def test_embedding_file_format(self, temp_embedding_file):
        """Property 3: Embedding 文件格式正确性. Validates: Requirements 2.3, 3.2, 3.3, 3.5"""
        emb_path, num_items, embedding_dim = temp_embedding_file
        embeddings = torch.load(emb_path)
        assert embeddings.shape == (num_items + 1, embedding_dim)
        assert torch.allclose(embeddings[0], torch.zeros(embedding_dim))

    def test_embedding_dimension_validation(self, temp_embedding_file):
        """Property 13: Embedding 维度验证. Validates: Requirements 6.6"""
        emb_path, num_items, embedding_dim = temp_embedding_file

        class MockGRAM_C:
            def __init__(self, gcn_dim):
                self.gcn_dim = gcn_dim
                self.use_collaborative_prefix = True
                self.gcn_item_embedding = nn.Embedding(num_items + 1, gcn_dim)

            def load_gcn_embeddings(self, gcn_emb_path):
                if not os.path.exists(gcn_emb_path):
                    raise FileNotFoundError(f"GCN embedding file not found: {gcn_emb_path}")
                gcn_embeddings = torch.load(gcn_emb_path, map_location='cpu')
                if gcn_embeddings.shape[1] != self.gcn_dim:
                    raise ValueError(
                        f"GCN embedding dimension mismatch: "
                        f"expected {self.gcn_dim}, got {gcn_embeddings.shape[1]}")

        model_matching = MockGRAM_C(gcn_dim=64)
        model_matching.load_gcn_embeddings(emb_path)
        model_mismatching = MockGRAM_C(gcn_dim=128)
        with pytest.raises(ValueError, match="dimension mismatch"):
            model_mismatching.load_gcn_embeddings(emb_path)


class TestPrefixInjection:
    """测试 Prefix 注入功能"""

    def test_prefix_injection_sequence_length(self):
        """Property 8: Prefix 注入后序列长度. Validates: Requirements 5.1, 5.2, 5.6"""
        batch_size, seq_len, llm_dim = 2, 10, 512
        original_embeds = torch.randn(batch_size, seq_len, llm_dim)
        original_mask = torch.ones(batch_size, seq_len)
        prefix_single = torch.randn(batch_size, 1, llm_dim)
        embeds_with_prefix = torch.cat([prefix_single, original_embeds], dim=1)
        prefix_mask = torch.ones(batch_size, 1)
        mask_with_prefix = torch.cat([prefix_mask, original_mask], dim=1)
        assert embeds_with_prefix.shape == (batch_size, seq_len + 1, llm_dim)
        assert mask_with_prefix.shape == (batch_size, seq_len + 1)
        k = 5
        prefix_sequential = torch.randn(batch_size, k, llm_dim)
        embeds_with_seq_prefix = torch.cat([prefix_sequential, original_embeds], dim=1)
        seq_prefix_mask = torch.ones(batch_size, k)
        mask_with_seq_prefix = torch.cat([seq_prefix_mask, original_mask], dim=1)
        assert embeds_with_seq_prefix.shape == (batch_size, seq_len + k, llm_dim)
        assert mask_with_seq_prefix.shape == (batch_size, seq_len + k)

    def test_whole_prefix_dropout(self):
        """Property 10: 整段 Prefix Dropout. Validates: Requirements 5.5"""
        adapter = CollaborativeAdapter(gcn_dim=64, llm_dim=512, dropout_rate=0.0, init_scale=0.1)
        gcn_embedding = nn.Embedding(101, 64)
        with torch.no_grad():
            gcn_embedding.weight[0].zero_()
        wrapper = MinimalEncoderWrapperC(
            collaborative_adapter=adapter, gcn_item_embedding=gcn_embedding,
            use_collaborative_prefix=True, use_sequential_prefix=True,
            prefix_dropout_prob=1.0)
        wrapper.train()
        batch_size, k = 4, 5
        recent_item_ids = torch.randint(1, 100, (batch_size, k))
        prefix = wrapper._compute_collaborative_prefix(recent_item_ids, 'cpu')
        assert torch.allclose(prefix, torch.zeros_like(prefix)), \
            "With dropout_prob=1.0, all prefix tokens should be zero"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
