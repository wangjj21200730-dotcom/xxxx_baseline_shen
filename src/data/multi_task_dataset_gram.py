import copy
import torch
import json
import os
import logging
from torch.utils.data import Dataset
from utils import indexing
import torch.distributed as dist

from time import time


class MultiTaskDatasetGRAM(Dataset):
    def __init__(
        self, args, dataset, mode, model_gen, tokenizer, phase=0, regenerate=False
    ):
        super().__init__()
        assert (
            regenerate == False
        ), "regenerate is not supported in MultiTaskDatasetGRAM. id is fixed"
        assert args.sample_num

        self.model_gen = model_gen
        self.tokenizer = tokenizer
        self.data_path = args.data_path
        self.dataset = dataset
        self.tasks = args.tasks.split(",")
        if args.sample_prompt > 0:
            assert len(self.tasks) == len(
                args.sample_num.split(",")
            ), "prompt sample number does not match task number"
        self.mode = mode
        self.args = args
        self.phase = phase

        self.rank = args.rank
        self.skip_empty_his = args.skip_empty_his
        self.reverse_history = args.reverse_history
        self.user_id_without_target_item = args.user_id_without_target_item
        self.id_linking = args.id_linking

        # GRAM-C: 协同前缀相关配置
        self.use_collaborative_prefix = getattr(args, 'use_collaborative_prefix', 0)
        self.recent_k = getattr(args, 'recent_k', 5)
        self.item_id_to_gcn_index = {}
        
        # 加载 ID 映射表
        if self.use_collaborative_prefix:
            self._load_item_id_mapping(args)

        if self.rank == 0:
            logging.info(f"Generating data for {self.dataset} dataset")

        self.max_his = args.max_his
        self.his_sep = args.his_sep

        # apply indexing method and avoid generate data multiple times
        if args.distributed:
            if self.rank == 0:
                logging.info("Reindex data with generative indexing method")
                indexing.gram_indexing(
                    data_path=self.data_path,
                    dataset=self.dataset,
                    model_gen=self.model_gen,
                    tokenizer=self.tokenizer,
                    regenerate=regenerate,
                    phase=self.phase,
                    args=self.args,
                    user_id_without_target_item=self.user_id_without_target_item,
                    id_linking=self.id_linking,
                )
                dist.barrier()
            else:
                dist.barrier()
            self.user_seq_dict, self.item2input, self.item2lexid = (
                indexing.gram_indexing(
                    data_path=self.data_path,
                    dataset=self.dataset,
                    model_gen=self.model_gen,
                    tokenizer=self.tokenizer,
                    regenerate=regenerate,
                    phase=self.phase,
                    args=self.args,
                    user_id_without_target_item=self.user_id_without_target_item,
                    id_linking=self.id_linking,
                )
            )
        else:
            logging.info("Reindex data with generative indexing method")
            self.user_seq_dict, self.item2input, self.item2lexid = (
                indexing.gram_indexing(
                    data_path=self.data_path,
                    dataset=self.dataset,
                    model_gen=self.model_gen,
                    tokenizer=self.tokenizer,
                    regenerate=regenerate,
                    phase=self.phase,
                    args=self.args,
                    user_id_without_target_item=self.user_id_without_target_item,
                    id_linking=self.id_linking,
                )
            )
        self.all_items = list(self.item2lexid.values())

        # load data
        if self.mode == "train":
            if self.rank == 0:
                logging.info("loading training data")
            self.data_samples = self.load_train()
        elif self.mode == "validation":
            self.data_samples = self.load_validation()
            if self.rank == 0:
                logging.info("loading validation data")
        else:
            raise NotImplementedError

        if self.args.debug_train_100:
            self.data_samples = self.data_samples[:100]
            if self.rank == 0:
                logging.info(
                    ">>>> Debug mode: only use 100 samples for training (MultiTaskDatasetGRAM)"
                )

        self.get_prompt_info()
        self.construct_sentence()

    def _load_item_id_mapping(self, args):
        """
        加载 raw_item_id 到 GCN index 的映射表
        
        映射表格式: Dict[str, int]
        - key: raw_item_id (如 "B00005N7P0")
        - value: gcn_index (1 ~ N, 0 保留给 unknown)
        """
        mapping_path = getattr(args, 'item_id_to_gcn_index_path', '')
        
        if not mapping_path:
            # 尝试默认路径
            mapping_path = os.path.join(
                args.data_path, self.dataset, 'item_id_to_gcn_index.json'
            )
        
        if os.path.exists(mapping_path):
            if mapping_path.endswith('.json'):
                with open(mapping_path, 'r') as f:
                    self.item_id_to_gcn_index = json.load(f)
            elif mapping_path.endswith('.pkl'):
                import pickle
                with open(mapping_path, 'rb') as f:
                    self.item_id_to_gcn_index = pickle.load(f)
            else:
                raise ValueError(f"Unsupported mapping file format: {mapping_path}")
            
            if self.rank == 0:
                logging.info(f"Loaded item_id to GCN index mapping from {mapping_path}")
                logging.info(f"  - Number of items in mapping: {len(self.item_id_to_gcn_index)}")
        else:
            if self.rank == 0:
                logging.warning(
                    f"Item ID mapping file not found: {mapping_path}. "
                    f"Collaborative prefix will use zero vectors for all items."
                )
            self.item_id_to_gcn_index = {}

    def _get_recent_item_gcn_indices(self, history_raw_ids):
        """
        获取最近 K 个物品的 GCN 索引
        
        Args:
            history_raw_ids: List[str] 历史物品的 raw_id 列表（已按时间顺序排列）
            
        Returns:
            List[int] 长度为 recent_k 的 GCN 索引列表，不足时前面补 0
        """
        # 取最近 K 个
        recent_ids = history_raw_ids[-self.recent_k:] if history_raw_ids else []
        
        # 转换为 GCN 索引（未知 ID 映射到 0）
        recent_gcn_indices = [
            self.item_id_to_gcn_index.get(raw_id, 0)
            for raw_id in recent_ids
        ]
        
        # Padding：不足 K 个时前面补 0
        if len(recent_gcn_indices) < self.recent_k:
            padding = [0] * (self.recent_k - len(recent_gcn_indices))
            recent_gcn_indices = padding + recent_gcn_indices
        
        return recent_gcn_indices

    def get_positive(self):
        """
        Get a dict of set to save the positive interactions for negative candidate sampling
        """
        positive = dict()
        for user in self.user_seq_dict:
            if self.mode == "train":
                positive[user] = set(self.user_seq_dict[user][:-2])
            if self.mode == "validation":
                positive[user] = set(self.user_seq_dict[user][:-1])
            if self.mode == "test":
                positive[user] = set(self.user_seq_dict[user])
        return positive

    def shuffle(self, seed):
        g = torch.Generator()
        g.manual_seed(seed)

        for task in self.task_data:
            indices = torch.randperm(len(self.task_data[task]), generator=g).tolist()
            self.task_data[task] = [self.task_data[task][i] for i in indices]

    def get_prompt_info(self):
        """
        Simplified version for single task with hardcoded prompts
        """
        # For single task, just create simple task_data mapping
        self.task_data = {self.tasks[0]: list(range(len(self.data_samples)))}
        # These might not be needed for single task, but keeping for compatibility
        self.task_prompt_num = [1]
        self.task_index = [len(self.data_samples)]

    def load_train(self):
        """
        Load training data samples

        already adopting data augmentation (for loop - range(len(items)))
        - Original: A B C D -> E
        - Augmented: A -> B, A B -> C, A B C -> D, A B C D -> E
        """
        st = time()
        data_samples = []
        for user in self.user_seq_dict:
            items = self.user_seq_dict[user][:-2]
            for i in range(len(items)):
                if i == 0:
                    if self.skip_empty_his > 0:
                        continue
                one_sample = dict()
                one_sample["dataset"] = self.dataset
                one_sample["user_id"] = user  # 'A1YJEY40YUW4SE'
                one_sample["target"] = items[i]  # 'B004ZT0SSG'
                one_sample["target_lex_id"] = self.item2lexid[
                    items[i]
                ]  # 'red shatter crackle nail polish e55'

                history = items[:i]
                if self.max_his > 0:
                    history = history[-self.max_his :]
                one_sample["history"] = self.his_sep.join(
                    history
                )  # 'B00KAL5JAU ; B00KHGIK54'
                one_sample["history_input"] = [
                    self.item2input[h] for h in history
                ]  # ['nail lacquer simmer and shimmer', ...]

                if self.reverse_history:
                    tmp_history = copy.deepcopy(one_sample["history"]).split(
                        self.his_sep
                    )
                    one_sample["history"] = self.his_sep.join(tmp_history[::-1])
                    one_sample["history_input"] = one_sample["history_input"][::-1]

                # TODO prefix hardcoded
                history_lex_ids = [
                    self.item2lexid[h] for h in history[::-1]
                ]  # reverse order
                one_sample["history_lex_id"] = self.his_sep.join(
                    history_lex_ids
                )  # 'nail lacquer simmer and shimmer ; red shatter crackle nail polish e55 ; ...'

                data_samples.append(one_sample)

        if self.rank == 0 and self.args.verbose_input_output:
            logging.info(f">> Load training data time: {time()-st} s")

        return data_samples

    def load_validation(self):
        """
        Load validation data samples
        """
        st = time()
        data_samples = []
        for user in self.user_seq_dict:
            items = self.user_seq_dict[user]
            one_sample = dict()
            one_sample["dataset"] = self.dataset
            one_sample["user_id"] = user
            one_sample["target"] = items[-2]
            one_sample["target_lex_id"] = self.item2lexid[items[-2]]

            history = items[:-2]
            if self.max_his > 0:
                history = history[-self.max_his :]
            one_sample["history"] = self.his_sep.join(history)
            one_sample["history_input"] = [self.item2input[h] for h in history]

            if self.reverse_history:
                tmp_history = copy.deepcopy(one_sample["history"]).split(self.his_sep)
                one_sample["history"] = self.his_sep.join(tmp_history[::-1])
                one_sample["history_input"] = one_sample["history_input"][::-1]

            history_lex_ids = [self.item2lexid[h] for h in history[::-1]]
            one_sample["history_lex_id"] = self.his_sep.join(
                history_lex_ids
            )  # 'nail lacquer simmer and shimmer ; red shatter crackle nail polish e55 ; ...'

            data_samples.append(one_sample)

        if self.rank == 0 and self.args.verbose_input_output:
            logging.info(f">> Load validation data time: {time()-st} s")

        return data_samples

    def construct_sentence(self):
        """
        Make data_samples into model input, output pairs
        data_samples: {
            'dataset': 'Beauty',
            'user_id': 'A2CG5Y82ZZNY6W',
            'target': 'B00KHH2VOY',
            'target_lex_id': 'dead sea salt soap body soap dead sea',
            'history': 'B00KAL5JAU ; B00KHGIK54',
            'history_input': ['dead sea salt deep hair conditioner natural curly',
                                'adovia facial serum anti aging'],
            'history_lex_id': 'dead sea salt deep hair conditioner natural curly ; adovia facial serum anti aging',
            }
        """

        st = time()
        self.data = {}
        self.data["input"] = []
        self.data["output"] = []
        self.data["user_id"] = []
        self.data["history_raw_ids"] = []  # GRAM-C: 保存历史物品的 raw_id
        
        for i in range(len(self.data_samples)):
            datapoint = self.data_samples[i]
            input_sample = []

            sequence_text = datapoint["history_lex_id"]
            user_sentence = f"What would user purchase after {sequence_text} ?"
            input_sample.append(user_sentence)

            history_input = datapoint["history_input"]
            input_sample += history_input

            self.data["input"].append(input_sample)
            self.data["output"].append(datapoint["target_lex_id"])
            self.data["user_id"].append(datapoint["user_id"])
            
            # GRAM-C: 保存历史物品的 raw_id（用于 GCN 查找）
            # history 格式: 'B00KAL5JAU ; B00KHGIK54'
            history_str = datapoint.get("history", "")
            if history_str:
                # 注意：如果 reverse_history=True，这里的顺序已经是反转后的
                # 我们需要原始时间顺序（最近的在最后）
                if self.reverse_history:
                    # 反转回原始顺序
                    history_raw_ids = history_str.split(self.his_sep)[::-1]
                else:
                    history_raw_ids = history_str.split(self.his_sep)
            else:
                history_raw_ids = []
            self.data["history_raw_ids"].append(history_raw_ids)

        if self.rank == 0 and self.args.verbose_input_output:
            logging.info(f">> Constructing sentence time: {time()-st} s")
            logging.info(
                f">> Input: {self.data['input'][-1]} \n>> Output: {self.data['output'][-1]}"
            )

    def __len__(self):
        return len(self.data["input"])

    def __getitem__(self, idx):
        result = {
            "input": self.data["input"][idx],
            "output": self.data["output"][idx],
            "user_id": self.data["user_id"][idx],
        }
        
        # GRAM-C: 添加 recent_item_ids
        if self.use_collaborative_prefix:
            history_raw_ids = self.data["history_raw_ids"][idx]
            recent_gcn_indices = self._get_recent_item_gcn_indices(history_raw_ids)
            result["recent_item_ids"] = torch.tensor(recent_gcn_indices, dtype=torch.long)
        
        return result
