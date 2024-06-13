# DPT baseline

This is DPT baseline. It uses deepspeed with ZeRO 2 for distributed training. To run, do

```commandline
export WORLD_SIZE=$(nvidia-smi -L | wc -l)
export XLA_PYTHON_CLIENT_PREALLOCATE=false

deepspeed --num_gpus "$WORLD_SIZE" train_xland.py \
    --config_path='configs/dpt_xland.yaml' \
    --learning_histories_path='<path_to_dataset>' \
    --seq_len=4096
```

To evaluate a model, run
```commandline
export WORLD_SIZE=$(nvidia-smi -L | wc -l)
export XLA_PYTHON_CLIENT_PREALLOCATE=false

deepspeed --num_gpus "$WORLD_SIZE" evaluate_in_context.py \
  --config_path='configs/dpt_xland.yaml' \
  --checkpoints_path='<path_to_model_folder>' \
  --learning_histories_path='<path_to_dataset>' \
  --eval_rulesets=1024 \
  --eval_episodes=500 \
  --seq_len=4096 \
```