# Conda 环境与服务器迁移

本文档用于把项目环境标准化到 Conda，并保证迁移到服务器后可复现运行。

## 1. 环境文件

- `envs/pro-h100.yml`:
  - 适合 H100 / CUDA 12.1 训练
  - 推荐用于你当前 4xH100 服务器
- `envs/pro-legacy.yml`:
  - 更接近项目原始依赖栈（torch 1.13 + cuda 11.7）
  - 用于兼容旧环境

## 2. 一键安装（推荐）

```bash
bash scripts/setup_conda_env.sh auto
```

参数说明：
- `auto`: 自动检测 GPU，H100 则选 `h100`，否则选 `legacy`
- 也可显式指定：
  - `bash scripts/setup_conda_env.sh h100`
  - `bash scripts/setup_conda_env.sh legacy`
- 自定义环境名：
  - `bash scripts/setup_conda_env.sh h100 pro-train`

安装结束会自动运行：
- `scripts/verify_runtime.py`

## 3. 手动安装（可选）

```bash
conda env create -f envs/pro-h100.yml -n pro-h100
conda activate pro-h100
python scripts/verify_runtime.py
```

## 4. 迁移到服务器

### 方案 A（推荐）：用同一份 yml 在服务器重建

把仓库同步到服务器后：

```bash
bash scripts/setup_conda_env.sh h100
```

### 方案 B：导出锁定环境再重建

在本地导出：

```bash
conda env export -n pro-h100 --no-builds > envs/pro-h100.lock.yml
```

在服务器导入：

```bash
conda env create -f envs/pro-h100.lock.yml -n pro-h100
conda run -n pro-h100 python scripts/verify_runtime.py
```

## 5. 常见问题

- `deepspeed` 编译慢：
  - 首次安装会编译 ops，属于正常
- `git-lfs` 不可用：
  - 运行 `git lfs install`
- CUDA 不可见：
  - 先检查 `nvidia-smi`
  - 再运行 `python scripts/verify_runtime.py` 查看 torch 是否识别 GPU
