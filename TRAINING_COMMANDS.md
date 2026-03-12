# PRO 训练指令清单

本文档汇总当前仓库可用的数据准备与训练命令，包含：
- HH-RLHF
- Summarize from Feedback
- ListUltraFeedback（`NDCG-alignment/ListUltraFeedback`，已改为直接使用多 response 数据）
- 其它数据集复用模板

## 1. 环境准备

```bash
pip install -r requirements.txt
```

如果你要在服务器稳定复现，优先使用 Conda：
- 说明文档：`CONDA_ENV.md`
- 一键安装：`bash scripts/setup_conda_env.sh auto`

## 2. GitHub 大文件（Git LFS）

项目已配置 `.gitattributes`，以下类型会自动走 LFS：
- 模型与权重：`*.pt`, `*.pth`, `*.bin`, `*.safetensors`, `*.ckpt`, `*.onnx`, `*.gguf`
- 数据与中间产物：`*.parquet`, `*.arrow`, `*.npy`, `*.npz`, `*.h5`, `*.hdf5`, `*.pkl`, `*.pickle`, `data/**/*.json`
- 压缩包与媒体：`*.zip`, `*.7z`, `*.tar`, `*.tar.gz`, `*.tgz`, `*.mp4`, `*.mov`, `*.mkv`, `*.webm`, `*.wav`
- 目录级：`checkpoints/**`, `transformers_cache/**`

首次机器初始化：

```bash
git lfs install
```

把已在 Git 历史中的大文件迁移到 LFS（如果你后续发现有 >100MB 历史文件）：

```bash
git lfs migrate import --include="*.pt,*.pth,*.bin,*.safetensors,*.ckpt,*.onnx,*.gguf,*.parquet,*.arrow,*.npy,*.npz,*.h5,*.hdf5,*.pkl,*.pickle,*.zip,*.7z,*.tar,*.tar.gz,*.tgz,*.mp4,*.mov,*.mkv,*.webm,*.wav,data/**/*.json,checkpoints/**,transformers_cache/**"
```

## 3. 4xH100 推荐配置（高吞吐）

当前 `train/*.sh` 已默认按 4 卡优化：
- `GPU_IDS=0,1,2,3`
- `NUM_PROCESSES=4`
- `ACCELERATE_CONFIG_FILE=ds_config_h100x4.yaml`
- `PER_DEVICE_TRAIN_BATCH_SIZE=4`
- `PER_DEVICE_EVAL_BATCH_SIZE=32`
- `CHECKPOINTING_STEP=2000`（减少中途验证造成的 GPU 空转）
- `PRO_DATALOADER_WORKERS=4`

如需手动调参，可在命令前覆写环境变量，例如：

```bash
cd train
GPU_IDS=0,1,2,3 \
NUM_PROCESSES=4 \
PER_DEVICE_TRAIN_BATCH_SIZE=6 \
PRO_DATALOADER_WORKERS=8 \
./train_ultrafeedback.sh <exp_id> 8 list_ultrafeedback_train_len8 list_ultrafeedback_dev_len8 list_ultrafeedback_test_len8
```

## 4. HH-RLHF

### 4.1 数据预处理

```bash
cd train/hh_preprocess_data
python step_1_process.py
python step_2_gen_train_data.py
python step_3_gen_test_data.py
cd ../..
```

### 4.2 训练

```bash
cd train
./train_hh.sh <exp_id> hh_train_len2 2
cd ..
```

参数说明：
- `exp_id`: 实验编号（日志和 checkpoint 目录会使用）
- `hh_train_len2`: 训练数据目录（位于 `data/` 下）
- `2`: 排序长度（`training_stage_num`）

## 5. Summarize from Feedback

### 5.1 数据预处理

```bash
cd train/summarize_preprocess_data
python step_1_process.py
python step_2_gen_train_data.py
python step_3_gen_test_data.py
cd ../..
```

### 5.2 训练

```bash
cd train
./train_summarize.sh <exp_id> summarize_train_len2 2
cd ..
```

如需长度 3 的配置，可用：

```bash
cd train
./train3_summarize.sh <exp_id> summarize_train_len3_alpaca 3
cd ..
```

## 6. ListUltraFeedback（直接多 response 训练）

### 6.1 数据准备与切分（train/dev/test）

```bash
cd train/list_ultrafeedback_preprocess_data
python step_1_prepare_data.py \
  --dataset_name NDCG-alignment/ListUltraFeedback \
  --ranking_len 8 \
  --split_strategy auto \
  --dev_ratio_from_train 0.05 \
  --train_dir list_ultrafeedback_train_len8 \
  --dev_dir list_ultrafeedback_dev_len8 \
  --test_dir list_ultrafeedback_test_len8
cd ../..
```

说明：
- `split_strategy=auto` 时，如果原数据集有 `test` split，会保留为测试集；再从 `train` 切一部分做验证集。
- 产物是 PRO 所需 JSONL 格式，分别写入：
  - `data/list_ultrafeedback_train_len8/train.json`
  - `data/list_ultrafeedback_dev_len8/dev.json`
  - `data/list_ultrafeedback_test_len8/test.json`

### 6.2 训练（含验证和最终测试打分）

```bash
cd train
./train_ultrafeedback.sh <exp_id> 8 \
  list_ultrafeedback_train_len8 \
  list_ultrafeedback_dev_len8 \
  list_ultrafeedback_test_len8
cd ..
```

训练日志会记录：
- Dev avg reward（训练中 checkpoint 验证）
- Final Test avg reward（训练结束后自动评估）

## 7. 其它数据集复用模板

若你的新数据集也是“一个 prompt + 多个候选回答 + 打分”，优先复用：
- `train/list_ultrafeedback_preprocess_data/step_1_prepare_data.py`

直接替换数据集名：

```bash
cd train/list_ultrafeedback_preprocess_data
python step_1_prepare_data.py \
  --dataset_name <your_hf_dataset_name> \
  --dataset_config <optional_config_name> \
  --ranking_len <N> \
  --train_dir <your_train_dir> \
  --dev_dir <your_dev_dir> \
  --test_dir <your_test_dir>
cd ../..
```

然后继续用：

```bash
cd train
./train_ultrafeedback.sh <exp_id> <N> <your_train_dir> <your_dev_dir> <your_test_dir>
cd ..
```

如果你的字段名不是 `prompt/completions/overall_scores`，修改：
- `train/list_ultrafeedback_preprocess_data/step_1_prepare_data.py` 中 `_build_pro_sample()` 的字段映射即可。
