export OMP_NUM_THREADS=16
root_dir=..

id=$1
ranking_len=$2

if [ -z "$id" ] || [ -z "$ranking_len" ]; then
    echo "Usage: $0 <exp_id> <ranking_len> [train_dir] [dev_dir] [test_dir]"
    exit 1
fi

train_dir=${3:-list_ultrafeedback_train_len${ranking_len}}
dev_dir=${4:-list_ultrafeedback_dev_len${ranking_len}}
test_dir=${5:-list_ultrafeedback_test_len${ranking_len}}

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
    --task ultrafeedback \
    --train_file_path "$root_dir/data/${train_dir}" \
    --validation_file_path "$root_dir/data/${dev_dir}" \
    --validation_file_name dev.json \
    --test_file_path "$root_dir/data/${test_dir}" \
    --test_file_name test.json \
    --output_dir "$root_dir/checkpoints/index_$id/stage_$ranking_len" \
    --log_path "$root_dir/logs/$id/$ranking_len" \
    --index "$id" \
    --seed 42 \
    --temperature 1 \
    --sft_weight 0.05 \
    --num_train_epochs "$num_train_epochs" \
    --checkpointing_step "$checkpointing_step" \
    --training_stage_num "$ranking_len" \
    --block_size 1024 \
    --learning_rate 5e-6 \
    --per_device_train_batch_size "$per_device_train_batch_size" \
    --per_device_eval_batch_size "$per_device_eval_batch_size" \
    --model_name_or_path decapoda-research/llama-7b-hf \
    --do_train \
    --do_validation > "$root_dir/logs/$id/$ranking_len/train_detail.log" 2>&1
