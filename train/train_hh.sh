export OMP_NUM_THREADS=16
root_dir=..

#stage 23
id=$1
data_path=$2
ranking_len=$3
if [ -z "$id" ] || [ -z "$data_path" ] || [ -z "$ranking_len" ]; then
    echo "Usage: $0 <exp_id> <data_path> <ranking_len>"
    exit 1
fi

gpu_ids=${GPU_IDS:-0,1,2,3}
num_processes=${NUM_PROCESSES:-4}
if [ -n "${ACCELERATE_CONFIG_FILE:-}" ]; then
    config_file="${ACCELERATE_CONFIG_FILE}"
elif command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | grep -Eiq "B100|B200|GB200|BLACKWELL|RTX 50|RTX50"; then
    config_file="ds_config_blackwellx4.yaml"
else
    config_file="ds_config_h100x4.yaml"
fi
per_device_train_batch_size=${PER_DEVICE_TRAIN_BATCH_SIZE:-4}
per_device_eval_batch_size=${PER_DEVICE_EVAL_BATCH_SIZE:-32}
checkpointing_step=${CHECKPOINTING_STEP:-2000}
num_train_epochs=${NUM_TRAIN_EPOCHS:-2}
dataloader_workers=${PRO_DATALOADER_WORKERS:-4}

mkdir -p "$root_dir/logs/$id/$ranking_len"
PRO_DATALOADER_WORKERS="$dataloader_workers" CUDA_VISIBLE_DEVICES="$gpu_ids" accelerate launch --num_processes "$num_processes" --config_file "$config_file" main.py \
    --task hh \
    --train_file_path "$root_dir/data/${data_path}" \
    --validation_file_path "$root_dir/data/hh_dev" \
    --validation_file_name sampled_dev.json \
    --output_dir "$root_dir/checkpoints/index_$id/stage_$ranking_len" \
    --log_path "$root_dir/logs/$id/$ranking_len" \
    --index "$id" \
    --seed 42 \
    --temperature 1 \
    --sft_weight 0.05 \
    --num_train_epochs "$num_train_epochs" \
    --checkpointing_step "$checkpointing_step" \
    --training_stage_num "$ranking_len" \
    --block_size 512 \
    --learning_rate 5e-6 \
    --per_device_train_batch_size "$per_device_train_batch_size" \
    --per_device_eval_batch_size "$per_device_eval_batch_size" \
    --model_name_or_path decapoda-research/llama-7b-hf \
    --do_train \
    --do_validation > "$root_dir/logs/$id/$ranking_len/train_detail.log" 2>&1
