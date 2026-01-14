# GPU 配置指南

## 快速参考

### 单卡训练（默认）

**只需修改 1 处**：
```bash
export CUDA_VISIBLE_DEVICES=0  # 改成你想用的 GPU ID
```

### 多卡训练

**需要修改 3 处**：
```bash
# 1. 设置可见 GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 2. 在 python 命令中添加
--distributed 1 \
--gpu 0,1,2,3 \
--master_port 12345 \
```

---

## 详细说明

### 情况 1: 单卡训练（当前 train_gram_c_beauty.sh）

#### 修改方法

只需修改脚本开头的这一行：

```bash
# 原始（使用 GPU 0）
export CUDA_VISIBLE_DEVICES=0

# 改成使用 GPU 1
export CUDA_VISIBLE_DEVICES=1

# 改成使用 GPU 2
export CUDA_VISIBLE_DEVICES=2
```

#### 完整示例

```bash
#!/bin/bash
# 设置环境变量
export CUDA_VISIBLE_DEVICES=1  # ← 只需改这里

# ... 其他配置不变 ...

# 运行训练（不需要 --distributed 参数）
python src/main_generative_gram.py \
    --datasets Beauty \
    --batch_size 16 \
    # ... 其他参数 ...
```

---

### 情况 2: 多卡训练（新建 train_gram_c_beauty_multi_gpu.sh）

#### 修改方法

需要修改 **3 个地方**：

```bash
# ============ 第 1 处：设置可见的 GPU ============
export CUDA_VISIBLE_DEVICES=0,1,2,3  # ← 改这里

# ============ 第 2 处：启用分布式 ============
DISTRIBUTED=1  # ← 改这里（0=单卡，1=多卡）

# ============ 第 3 处：指定 GPU 列表 ============
GPU_LIST="0,1,2,3"  # ← 改这里（必须与 CUDA_VISIBLE_DEVICES 一致）

# 运行训练
python src/main_generative_gram.py \
    --distributed ${DISTRIBUTED} \
    --gpu ${GPU_LIST} \
    --master_port 12345 \
    # ... 其他参数 ...
```

#### 完整示例（使用 4 张卡）

```bash
#!/bin/bash
# GPU 配置
export CUDA_VISIBLE_DEVICES=0,1,2,3  # 使用 GPU 0,1,2,3
DISTRIBUTED=1
GPU_LIST="0,1,2,3"
MASTER_PORT=12345

# 训练配置（多卡可以用更大的 batch size）
BATCH_SIZE=32  # 单卡是 16，多卡可以翻倍

# 运行训练
python src/main_generative_gram.py \
    --datasets Beauty \
    --distributed ${DISTRIBUTED} \
    --gpu ${GPU_LIST} \
    --master_port ${MASTER_PORT} \
    --batch_size ${BATCH_SIZE} \
    # ... 其他参数 ...
```

---

## 常见配置示例

### 示例 1: 使用 GPU 0（单卡）

```bash
export CUDA_VISIBLE_DEVICES=0
# 不需要 --distributed 参数
```

### 示例 2: 使用 GPU 1（单卡）

```bash
export CUDA_VISIBLE_DEVICES=1
# 不需要 --distributed 参数
```

### 示例 3: 使用 GPU 0,1（双卡）

```bash
export CUDA_VISIBLE_DEVICES=0,1
DISTRIBUTED=1
GPU_LIST="0,1"

python src/main_generative_gram.py \
    --distributed 1 \
    --gpu 0,1 \
    --master_port 12345 \
    # ...
```

### 示例 4: 使用 GPU 2,3（双卡）

```bash
export CUDA_VISIBLE_DEVICES=2,3
DISTRIBUTED=1
GPU_LIST="0,1"  # ← 注意：这里是逻辑 ID，从 0 开始

python src/main_generative_gram.py \
    --distributed 1 \
    --gpu 0,1 \  # ← 逻辑 ID
    --master_port 12345 \
    # ...
```

**重要**: 当 `CUDA_VISIBLE_DEVICES=2,3` 时，程序内部看到的是 GPU 0,1（逻辑 ID）

### 示例 5: 使用 GPU 0,1,2,3（四卡）

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
DISTRIBUTED=1
GPU_LIST="0,1,2,3"

python src/main_generative_gram.py \
    --distributed 1 \
    --gpu 0,1,2,3 \
    --master_port 12345 \
    # ...
```

---

## 参数说明

### CUDA_VISIBLE_DEVICES

- **作用**: 限制程序可见的物理 GPU
- **格式**: 逗号分隔的 GPU ID（物理 ID）
- **示例**: 
  - `0` - 只使用物理 GPU 0
  - `0,1` - 使用物理 GPU 0 和 1
  - `2,3` - 使用物理 GPU 2 和 3

### --distributed

- **作用**: 是否启用分布式训练
- **取值**: 
  - `0` - 单卡训练（默认）
  - `1` - 多卡训练

### --gpu

- **作用**: 指定使用的 GPU 列表（逻辑 ID）
- **格式**: 逗号分隔的字符串
- **示例**: `"0,1,2,3"`
- **注意**: 这是逻辑 ID，从 0 开始编号

### --master_port

- **作用**: 分布式训练的通信端口
- **默认值**: 12345
- **何时修改**: 如果端口被占用，改成其他值（如 12346, 23456）

---

## Batch Size 调整建议

多卡训练时，可以按比例增大 batch size：

| GPU 数量 | 推荐 Batch Size | 说明 |
|---------|----------------|------|
| 1 卡 | 16 | 基准值 |
| 2 卡 | 32 | 2x |
| 4 卡 | 64 | 4x |
| 8 卡 | 128 | 8x |

**注意**: 如果显存不足，可以减小 batch size 并增大 `gradient_accumulation_steps`

---

## 常见问题

### Q1: 我有 4 张卡，但只想用其中 2 张，怎么办？

**A**: 指定你想用的卡：

```bash
# 使用物理 GPU 0 和 2
export CUDA_VISIBLE_DEVICES=0,2
DISTRIBUTED=1
GPU_LIST="0,1"  # 逻辑 ID

python src/main_generative_gram.py \
    --distributed 1 \
    --gpu 0,1 \
    # ...
```

### Q2: 端口被占用怎么办？

**A**: 改成其他端口：

```bash
--master_port 23456  # 或其他未被占用的端口
```

### Q3: 多卡训练时 CUDA Out of Memory？

**A**: 减小 batch size 或增加梯度累积：

```bash
BATCH_SIZE=16  # 减小
GRADIENT_ACCUMULATION_STEPS=4  # 增大
```

### Q4: 单卡训练需要设置 --distributed 吗？

**A**: 不需要。单卡训练时：
- 不设置 `--distributed`（或设为 0）
- 不设置 `--gpu`
- 不设置 `--master_port`

### Q5: CUDA_VISIBLE_DEVICES 和 --gpu 的区别？

**A**: 
- `CUDA_VISIBLE_DEVICES`: 物理 GPU ID（系统级别）
- `--gpu`: 逻辑 GPU ID（程序级别，从 0 开始）

**示例**:
```bash
export CUDA_VISIBLE_DEVICES=2,3  # 物理 GPU 2,3
--gpu 0,1  # 程序内部看到的是逻辑 GPU 0,1
```

---

## 快速检查命令

### 查看可用 GPU

```bash
nvidia-smi
```

### 查看当前进程占用的 GPU

```bash
nvidia-smi | grep python
```

### 测试 GPU 配置

```bash
# 单卡测试
CUDA_VISIBLE_DEVICES=0 python -c "import torch; print(torch.cuda.device_count())"
# 输出: 1

# 多卡测试
CUDA_VISIBLE_DEVICES=0,1,2,3 python -c "import torch; print(torch.cuda.device_count())"
# 输出: 4
```

---

## 推荐配置

### 开发/调试（单卡）

```bash
export CUDA_VISIBLE_DEVICES=0
BATCH_SIZE=8
REC_EPOCHS=1
--debug_train_100 1
```

### 正式训练（单卡）

```bash
export CUDA_VISIBLE_DEVICES=0
BATCH_SIZE=16
REC_EPOCHS=10
```

### 正式训练（多卡）

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
DISTRIBUTED=1
GPU_LIST="0,1,2,3"
BATCH_SIZE=64
REC_EPOCHS=10
```

---

## 总结

### 单卡训练

✅ **只需修改 1 处**: `export CUDA_VISIBLE_DEVICES=X`

### 多卡训练

✅ **需要修改 3 处**:
1. `export CUDA_VISIBLE_DEVICES=X,Y,Z`
2. `DISTRIBUTED=1`
3. `GPU_LIST="0,1,2"` + `--distributed 1 --gpu 0,1,2`

---

**最后更新**: 2026-01-14
