"""
GRAM / GRAM-C 主入口

支持:
- 原始 GRAM 模型
- GRAM-C (Collaborative-Enhanced GRAM) 模型
- 本地模型路径加载（离线运行）
"""

import torch
import os
import logging
import datetime
import warnings
import torch.distributed as dist
import torch.multiprocessing as mp

from IPython import embed
from transformers import T5Config, AutoTokenizer, AutoModelForSeq2SeqLM
from types import MethodType
from undecorated import undecorated

from runner import get_runner
from arguments import create_parser
from model import create_model
from utils import (
    set_seed,
    setup_logging,
    setup_model_path,
    save_args,
    load_model,
    get_dataset_gram,
    get_loader_gram,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings(action="ignore")


def get_local_model_path(args, model_name):
    """
    获取本地模型路径
    
    Args:
        args: 参数对象
        model_name: 模型名称 (如 "t5-small")
        
    Returns:
        本地模型路径
    """
    local_model_dir = getattr(args, 'local_model_dir', 'models/')

    if os.path.exists(model_name):
        return model_name
    
    # 处理不同的模型名称格式
    if model_name == "t5-small":
        local_path = os.path.join(local_model_dir, "t5_small")
    elif "t5-small-machine-articles-tag-generation" in model_name:
        local_path = os.path.join(local_model_dir, "t5-small-machine-articles-tag-generation")
    else:
        # 默认：将 - 替换为 _
        local_path = os.path.join(local_model_dir, model_name.replace("-", "_"))
    
    # 检查本地路径是否存在
    if os.path.exists(local_path):
        return local_path
    else:
        raise FileNotFoundError(
            f"Local model path not found: {local_path}. "
            f"Please download/copy the model to local_model_dir={local_model_dir}."
        )


def setup_gram_c_config(config, args):
    """
    设置 GRAM-C 相关的配置参数
    
    Args:
        config: T5Config 对象
        args: 参数对象
    """
    # GRAM-C 配置
    config.use_collaborative_prefix = getattr(args, 'use_collaborative_prefix', 0)
    config.gcn_dim = getattr(args, 'gcn_dim', 64)
    config.prefix_init_scale = getattr(args, 'prefix_init_scale', 0.1)
    config.prefix_dropout_prob = getattr(args, 'prefix_dropout_prob', 0.2)
    config.adapter_dropout = getattr(args, 'adapter_dropout', 0.1)
    config.recent_k = getattr(args, 'recent_k', 5)
    
    # 估算物品数量（后续可以从数据集获取精确值）
    config.num_items = getattr(args, 'num_items', 50000)
    
    return config


def distributed_launch(args):
    ngpus_per_node = torch.cuda.device_count()
    args.world_size = ngpus_per_node

    mp.spawn(distributed_main, args=(args,), nprocs=ngpus_per_node, join=True)


def distributed_main(local_rank, args):
    args.rank = local_rank
    set_seed(args.seed)
    os.environ["MASTER_ADDR"] = args.master_addr
    os.environ["MASTER_PORT"] = args.master_port
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        timeout=datetime.timedelta(seconds=18000),
        world_size=args.world_size,
        rank=local_rank,
    )
    setup_logging(args)
    setup_model_path(args)

    if args.rank == 0:
        save_args(args)
        logging.info(vars(args))

    device = f"cuda:{local_rank}"
    args.gpu = local_rank

    # 获取本地模型路径
    backbone_path = get_local_model_path(args, args.backbone)
    
    if "t5" in args.backbone:
        config = T5Config.from_pretrained(backbone_path)
        if local_rank == 0:
            logging.info(f"Use {backbone_path} backbone model")
    else:
        raise NotImplementedError

    config.max_seq_len = args.item_prompt_max_len
    config.max_item_num = args.max_his
    config.use_position_embedding = args.use_position_embedding
    config.sample_num = args.sample_num
    
    # 设置 GRAM-C 配置
    config = setup_gram_c_config(config, args)

    model_backbone = AutoModelForSeq2SeqLM.from_pretrained(backbone_path, config=config)
    
    # 根据配置选择模型类型
    use_collaborative_prefix = getattr(args, 'use_collaborative_prefix', 0)
    model_type = "gram_c" if use_collaborative_prefix else "gram"
    
    model_rec = create_model(model_type, config=config)
    model_rec.load_t5(model_backbone.state_dict())

    if local_rank == 0 and use_collaborative_prefix:
        logging.info(
            f"[GRAM-C DIAG] encoder wrapper type after load_t5: {type(model_rec.encoder)}"
        )
        logging.info(
            f"[GRAM-C DIAG] encoder has set_recent_item_ids: {hasattr(model_rec.encoder, 'set_recent_item_ids')}"
        )
    
    # 如果使用 GRAM-C，加载 GCN embeddings
    if use_collaborative_prefix and hasattr(model_rec, 'load_gcn_embeddings'):
        gcn_emb_path = getattr(args, 'gcn_item_emb_path', '')
        if gcn_emb_path and os.path.exists(gcn_emb_path):
            model_rec.load_gcn_embeddings(gcn_emb_path)
            if local_rank == 0:
                logging.info(f"Loaded GCN embeddings from {gcn_emb_path}")
        else:
            if local_rank == 0:
                logging.warning(f"GCN embedding file not found: {gcn_emb_path}")

    generate_with_grad = undecorated(model_rec.generate)
    model_rec.generate_with_grad = MethodType(generate_with_grad, model_rec)
    model_rec.to(device)

    # 加载 tag generation 模型（使用本地路径）
    tag_gen_path = get_local_model_path(args, "nandakishormpai/t5-small-machine-articles-tag-generation")
    model_gen = AutoModelForSeq2SeqLM.from_pretrained(tag_gen_path)
    generate_with_grad = undecorated(model_gen.generate)
    model_gen.generate_with_grad = MethodType(generate_with_grad, model_gen)
    model_gen.to(device)

    # 加载 tokenizer（使用本地路径）
    tokenizer = AutoTokenizer.from_pretrained(backbone_path)
    args.tokenizer = tokenizer

    if args.rec_model_path:
        if local_rank == 0:
            logging.info(f"Load model from {args.rec_model_path}")
        model_rec = load_model(model_rec, args.rec_model_path, args, loc=device)
        model_rec.to(device)

    if args.id_model_path:
        if local_rank == 0:
            logging.info(f"Load model from {args.id_model_path}")
        model_gen = load_model(model_gen, args.id_model_path, args, loc=device)
        model_gen.to(device)

    TrainSetID, TrainSetRec, ValidSet = get_dataset_gram(args, model_gen, tokenizer)
    train_loader_id, train_loader_rec, valid_loader = get_loader_gram(
        args, tokenizer, TrainSetID, TrainSetRec, ValidSet, local_rank
    )

    runner = get_runner(
        "distributed",
        model_rec,
        model_gen,
        tokenizer,
        train_loader_id,
        train_loader_rec,
        valid_loader,
        device,
        args,
        local_rank,
    )

    if args.train:
        if local_rank == 0:
            logging.info("Start training")
        runner.train_generator()

        if local_rank == 0:
            logging.info(f"Train done ... Load model from {runner.cur_model_path}")
        runner.test(runner.cur_model_path)
    dist.barrier()
    dist.destroy_process_group()

    return


def single_main(args):
    setup_logging(args)
    setup_model_path(args)
    set_seed(args.seed)

    args.rank = 0
    device = torch.device("cuda", int(args.gpu.split(",")[0]))

    save_args(args)
    logging.info(vars(args))
    
    # 获取本地模型路径
    backbone_path = get_local_model_path(args, args.backbone)
    
    # 加载 tokenizer（使用本地路径）
    tokenizer = AutoTokenizer.from_pretrained(backbone_path)

    if "t5" in args.backbone:
        config = T5Config.from_pretrained(backbone_path)
        logging.info(f"Use {backbone_path} backbone model")
    else:
        raise NotImplementedError

    config.max_seq_len = args.item_prompt_max_len
    config.max_item_num = args.max_his
    config.use_position_embedding = args.use_position_embedding
    config.sample_num = args.sample_num
    
    # 设置 GRAM-C 配置
    config = setup_gram_c_config(config, args)

    model_backbone = AutoModelForSeq2SeqLM.from_pretrained(backbone_path, config=config)
    
    # 根据配置选择模型类型
    use_collaborative_prefix = getattr(args, 'use_collaborative_prefix', 0)
    model_type = "gram_c" if use_collaborative_prefix else "gram"
    
    model_rec = create_model(model_type, config=config)
    model_rec.load_t5(model_backbone.state_dict())
    
    # 如果使用 GRAM-C，加载 GCN embeddings
    if use_collaborative_prefix and hasattr(model_rec, 'load_gcn_embeddings'):
        gcn_emb_path = getattr(args, 'gcn_item_emb_path', '')
        if gcn_emb_path and os.path.exists(gcn_emb_path):
            model_rec.load_gcn_embeddings(gcn_emb_path)
            logging.info(f"Loaded GCN embeddings from {gcn_emb_path}")
        else:
            logging.warning(f"GCN embedding file not found: {gcn_emb_path}")

    generate_with_grad = undecorated(model_rec.generate)
    model_rec.generate_with_grad = MethodType(generate_with_grad, model_rec)
    model_rec.to(device)

    # 加载 tag generation 模型（使用本地路径）
    tag_gen_path = get_local_model_path(args, "nandakishormpai/t5-small-machine-articles-tag-generation")
    model_gen = AutoModelForSeq2SeqLM.from_pretrained(tag_gen_path)
    generate_with_grad = undecorated(model_gen.generate)
    model_gen.generate_with_grad = MethodType(generate_with_grad, model_gen)
    model_gen.to(device)

    tokenizer = AutoTokenizer.from_pretrained(backbone_path)
    args.tokenizer = tokenizer

    TrainSetID, TrainSetRec, ValidSet = get_dataset_gram(
        args, model_gen, tokenizer, phase=0, regenerate=False
    )
    train_loader_id, train_loader_rec, valid_loader = get_loader_gram(
        args, tokenizer, TrainSetID, TrainSetRec, ValidSet
    )

    if args.rec_model_path:
        model_rec = load_model(model_rec, args.rec_model_path, args, loc=device)
        logging.info(f"Load recommender model from {args.rec_model_path}")

    if args.id_model_path:
        model_gen = load_model(model_gen, args.id_model_path, args, loc=device)
        logging.info(f"Load generator model from {args.id_model_path}")

    runner = get_runner(
        "single",
        model_rec,
        model_gen,
        tokenizer,
        train_loader_id,
        train_loader_rec,
        valid_loader,
        device,
        args,
    )

    if args.train:
        logging.info("Start training")
        runner.train_generator()

        logging.info(f"Train done ... Load model from {runner.cur_model_path}")
        runner.args.debug_test_100 = 0
        runner.test(runner.cur_model_path)
    else:
        if args.test_by_valid:
            logging.info(
                f"[VALID] Test model from {args.rec_model_path}, {args.id_model_path}"
            )
            runner.validate_from_model(args.rec_model_path, args.id_model_path)
        else:
            logging.info(f"Test model from {args.rec_model_path}, {args.id_model_path}")
            runner.test_from_model(args.rec_model_path, args.id_model_path)


if __name__ == "__main__":
    parser = create_parser()
    init_args, _ = parser.parse_known_args()
    ngpus_per_node = torch.cuda.device_count()
    if init_args.distributed and ngpus_per_node > 1:
        distributed_launch(args=init_args)
    else:
        single_main(args=init_args)
