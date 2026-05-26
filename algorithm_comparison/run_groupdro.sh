#!/bin/bash
set -e

# Minimal run script: baseline GRPO vs Prompt-GDRO (Problem 1)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VERL_DIR="$(dirname "$SCRIPT_DIR")"

# CLI args
GPU_SET=${1:-0}
CASE=${2:-grpo}  # grpo | gdro_passk_s8_hyst_norm_3bins_dapo | gdro_passk_s8_hyst_norm_6bins_dapo | gdro_passk_s8_hyst_norm_hardf_6bins_dapo | gdro_passk_s8_hyst_norm_softf_sched_6bins_noextremes_dapo | all
DATASET=${3:-gsm8k}  # gsm8k | math
MODEL_SIZE=${4:-1.7b-base}  # 1.7b-base | 4b-base | 8b-base

# Freeze experiment timestamp so delayed child invocations share the same directory
export EXP_STAMP=${EXP_STAMP:-$(date +%Y%m%d_%H%M%S)}

# Dataset selector
if [[ "$DATASET" == "gsm8k" ]]; then
  TRAIN_FILE="$VERL_DIR/data/gsm8k/train.parquet"
  VAL_FILE="$VERL_DIR/data/gsm8k/test.parquet"
  GDRO_CLASSIFIER="gsm8k"
elif [[ "$DATASET" == "math" ]]; then
  TRAIN_FILE="$VERL_DIR/data/math/train.parquet"
  VAL_FILE="$VERL_DIR/data/math/test.parquet"
  GDRO_CLASSIFIER="math"
elif [[ "$DATASET" == "math-dapo" ]]; then
  TRAIN_FILE="$VERL_DIR/data/math-dapo-style/train.parquet"
  # Compose a Hydra list of all benchmark parquets under math-dapo-style
  p1="$VERL_DIR/data/math-dapo-style/AIME-2024/test.parquet"
  p2="$VERL_DIR/data/math-dapo-style/AIME-2025/test.parquet"
  p3="$VERL_DIR/data/math-dapo-style/AMC/test.parquet"
  p4="$VERL_DIR/data/math-dapo-style/MATH-500/test.parquet"
  p5="$VERL_DIR/data/math-dapo-style/Minerva/test.parquet"
  p6="$VERL_DIR/data/math-dapo-style/OlympiadBench/test.parquet"
  p7="$VERL_DIR/data/math-dapo-style/GPQA/test.parquet"
  VAL_FILE='["'"$p1"'","'"$p2"'","'"$p3"'","'"$p4"'","'"$p5"'","'"$p6"'","'"$p7"'"]'
  GDRO_CLASSIFIER="math"
elif [[ "$DATASET" == "math-div" ]]; then
  TRAIN_FILE="$VERL_DIR/data/math/train.parquet"
  # Compose a Hydra list of all benchmark parquets under math-dapo-style
  p1="$VERL_DIR/data/math-dapo-style/AIME-2024/test.parquet"
  p2="$VERL_DIR/data/math-dapo-style/AIME-2025/test.parquet"
  p3="$VERL_DIR/data/math-dapo-style/AMC/test.parquet"
  p4="$VERL_DIR/data/math-dapo-style/MATH-500/test.parquet"
  p5="$VERL_DIR/data/math-dapo-style/Minerva/test.parquet"
  p6="$VERL_DIR/data/math-dapo-style/OlympiadBench/test.parquet"
  p7="$VERL_DIR/data/math-dapo-style/GPQA/test.parquet"
  VAL_FILE='["'"$p1"'","'"$p2"'","'"$p3"'","'"$p4"'","'"$p5"'","'"$p6"'","'"$p7"'"]'
  GDRO_CLASSIFIER="math"
else
  echo "Unknown DATASET=$DATASET. Supported: gsm8k | math | math-dapo | math-div"; exit 1
fi

EXP_DIR="$SCRIPT_DIR/results/prompt_gdro_${DATASET}_$EXP_STAMP"
mkdir -p "$EXP_DIR/logs" "$EXP_DIR/checkpoints" "$EXP_DIR/configs"

if [[ "$MODEL_SIZE" == "1.7b-base" ]]; then
  MODEL_PATH="Qwen/Qwen3-1.7B-Base"
elif [[ "$MODEL_SIZE" == "4b-base" ]]; then
  MODEL_PATH="Qwen/Qwen3-4B-Base"
elif [[ "$MODEL_SIZE" == "8b-base" ]]; then
  MODEL_PATH="Qwen/Qwen3-8B-Base"
else
  echo "Unknown MODEL_SIZE=$MODEL_SIZE. Supported: 1.7b-base | 4b-base | 8b-base"; exit 1
fi

if [[ ! -f "$TRAIN_FILE" ]]; then
  echo "Missing $TRAIN_FILE for dataset=$DATASET. Run data prep."; exit 1
fi

run_case() {
  local name=$1
  local gpus=$2
  local extra_cfg=$3

  local log_file="$EXP_DIR/logs/${name}.log"
  local ckpt_dir="$EXP_DIR/checkpoints/${name}"
  mkdir -p "$EXP_DIR/wandb/$name"

  # Support multi-GPU per case via '+' (e.g., 0+1). Translate to CUDA comma list.
  local gpu_env
  gpu_env=$(echo "$gpus" | tr '+' ',')

  # Dataset-specific max lengths (match run_knapsack.sh style)
  local length_args=""
  if [[ "$TRAIN_FILE" == "$VERL_DIR/data/math-dapo-style/train.parquet" ]]; then
    length_args="data.max_prompt_length=3200 data.max_response_length=4096"
  elif [[ "$DATASET" == "math" || "$DATASET" == "math-div" ]]; then
    length_args="data.max_prompt_length=2048 data.max_response_length=2560"
  else
    length_args="data.max_prompt_length=512 data.max_response_length=512"
  fi

  WANDB_DIR="$EXP_DIR/wandb/$name" CUDA_VISIBLE_DEVICES=$gpu_env python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    +algorithm.adv_clip_low=-5 \
    +algorithm.adv_clip_high=5 \
    algorithm.norm_adv_by_std_in_grpo=True \
    data.train_files=$TRAIN_FILE \
    data.val_files=$VAL_FILE \
    data.train_batch_size=512 \
    $length_args \
    data.dataloader_num_workers=0 \
    data.val_batch_size=512 \
    +algorithm.eval_group_metrics_enable=True \
    +algorithm.eval_group_key=math \
    +algorithm.gdro_apply_weights=True \
    $extra_cfg \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.val_kwargs.n=16 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    trainer.logger='["console", "wandb"]' \
    trainer.project_name="Prompt-GDRO" \
    trainer.experiment_name=${name}_${DATASET}_${MODEL_SIZE}_${EXP_STAMP} \
    trainer.n_gpus_per_node=$(echo "$gpus" | tr '+' '\n' | wc -l) \
    trainer.nnodes=1 \
    trainer.save_freq=0 \
    trainer.test_freq=40 \
    +trainer.total_training_steps=500 \
    trainer.default_local_dir=$ckpt_dir \
    trainer.log_val_generations=1 \
    trainer.val_before_train=False \
    +reward_model.num_examine_val=1 2>&1 | tee "$log_file"
}

# Cases

# If CASE and GPU_SET are comma-separated lists of equal length, run in parallel per GPU
if [[ "$CASE" == *","* ]]; then
  IFS="," read -r -a CASE_ARR <<< "$CASE"
  IFS="," read -r -a GPU_ARR <<< "$GPU_SET"
  if [[ ${#CASE_ARR[@]} -ne ${#GPU_ARR[@]} ]]; then
    echo "Error: number of cases and GPUs must match (cases=${#CASE_ARR[@]} gpus=${#GPU_ARR[@]})."; exit 1
  fi
  pids=()
  for i in "${!CASE_ARR[@]}"; do
    c=${CASE_ARR[$i]}
    g=${GPU_ARR[$i]}
    echo "Starting case $c on GPU $g (dataset=$DATASET)"
    # Stagger launches by 5 minutes per index to avoid 429s
    ( sleep $(( i * 300 )); CUDA_VISIBLE_DEVICES=$(echo "$g" | tr '+' ',') bash "$0" "$g" "$c" "$DATASET" "$MODEL_SIZE" ) &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
  echo "Parallel cases completed."; exit 0
fi

if [[ "$CASE" == "grpo" ]]; then
  # Plain GRPO baseline
  run_case grpo "$GPU_SET" ""
elif [[ "$CASE" == "gdro_passk_s8_hyst_norm_3bins_dapo" ]]; then
  run_case gdro_passk_s8_hyst_norm_3bins_dapo "$GPU_SET" \
    "+algorithm.gdro_enable=True +algorithm.gdro_weight_mode=class +algorithm.gdro_prompt_classifier=passk_online +algorithm.passk_edges='0.4,0.75' +algorithm.passk_exclude_extremes=True +algorithm.passk_history_len=50 +algorithm.passk_hysteresis=0.03 +algorithm.passk_num_bins=3 +algorithm.loss_norm_by_class=True +algorithm.gdro_debias_scores_ema=True +algorithm.gdro_ema_beta=0.12 +algorithm.gdro_eta_q=0.65 +algorithm.gdro_gamma=0.01 +algorithm.gdro_max_class_weight=15.0 algorithm.norm_adv_by_std_in_grpo=True"
elif [[ "$CASE" == "gdro_passk_s8_hyst_norm_6bins_dapo" ]]; then
  run_case gdro_passk_s8_hyst_norm_6bins_dapo "$GPU_SET" \
    "+algorithm.gdro_enable=True +algorithm.gdro_weight_mode=class +algorithm.gdro_prompt_classifier=passk_online +algorithm.passk_edges='0.1,0.2,0.4,0.6,0.8,0.9' +algorithm.passk_history_len=100 +algorithm.passk_hysteresis=0.03 +algorithm.passk_num_bins=6 +algorithm.loss_norm_by_class=True +algorithm.gdro_debias_scores_ema=True +algorithm.gdro_ema_beta=0.12 +algorithm.gdro_eta_q=0.65 +algorithm.gdro_gamma=0.01 +algorithm.gdro_max_class_weight=15.0 algorithm.norm_adv_by_std_in_grpo=True"
elif [[ "$CASE" == "gdro_passk_s8_hyst_norm_hardf_6bins_dapo" ]]; then
  run_case gdro_passk_s8_hyst_norm_hardf_6bins_dapo "$GPU_SET" \
    "+algorithm.gdro_enable=True +algorithm.gdro_weight_mode=class +algorithm.gdro_prompt_classifier=passk_online +algorithm.passk_edges='0.1,0.2,0.4,0.6,0.8,0.9' +algorithm.passk_history_len=100 +algorithm.passk_hysteresis=0.03 +algorithm.passk_num_bins=6 +algorithm.loss_norm_by_class=True +algorithm.passk_focus_enable=True +algorithm.passk_focus_warmup_steps=50 +algorithm.passk_focus_ramp_steps=200 +algorithm.passk_focus_map='{0:0.0,6:0.0}' +algorithm.gdro_debias_scores_ema=True +algorithm.gdro_ema_beta=0.12 +algorithm.gdro_eta_q=0.65 +algorithm.gdro_gamma=0.01 +algorithm.gdro_max_class_weight=15.0 algorithm.norm_adv_by_std_in_grpo=True"
elif [[ "$CASE" == "gdro_passk_s8_hyst_norm_softf_sched_6bins_noextremes_dapo" ]]; then
  run_case gdro_passk_s8_hyst_norm_softf_sched_6bins_noextremes_dapo "$GPU_SET" \
    "+algorithm.gdro_enable=True +algorithm.gdro_weight_mode=class +algorithm.gdro_prompt_classifier=passk_online +algorithm.passk_edges='0.01,0.2,0.4,0.6,0.8,0.99' +algorithm.passk_exclude_extremes=True +algorithm.passk_history_len=100 +algorithm.passk_hysteresis=0.03 +algorithm.passk_num_bins=6 +algorithm.loss_norm_by_class=True +algorithm.passk_focus_enable=True +algorithm.passk_focus_warmup_steps=50 +algorithm.passk_focus_ramp_steps=200 +algorithm.passk_focus_map='{0:0.0,1:0.9,2:0.8,3:0.7,4:0.7,5:0.5,6:0.0}' +algorithm.gdro_debias_scores_ema=True +algorithm.gdro_ema_beta=0.12 +algorithm.gdro_eta_q=0.65 +algorithm.gdro_gamma=0.01 +algorithm.gdro_max_class_weight=15.0 algorithm.norm_adv_by_std_in_grpo=True"
else
  # all
  run_case grpo "$GPU_SET" ""
fi

echo "Prompt-GDRO runs completed. Logs at $EXP_DIR/logs"


