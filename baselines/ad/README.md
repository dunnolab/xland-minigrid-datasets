# AD baseline

This is AD baseline. It uses deepspeed with ZeRO 2 for distributed training. To run, do

```commandline
export WORLD_SIZE=$(nvidia-smi -L | wc -l)

deepspeed --num_gpus "$WORLD_SIZE" train_xland.py \
    --config_path='configs/xland.yaml' \
    --learning_histories_path='<path_to_dataset>'
```

To evaluate a model, run
```commandline
export WORLD_SIZE=$(nvidia-smi -L | wc -l)

deepspeed --num_gpus "$WORLD_SIZE" evaluate_in_context.py \
  --config_path='<path_to_model_folder>' \
  --eval_rulesets=1024 \
  --eval_episodes=500
```

## References

Laskin, M., Wang, L., Oh, J., Parisotto, E., Spencer, S., Steigerwald, R., ... & Mnih, V. (2022). In-context reinforcement learning with algorithm distillation. arXiv preprint arXiv:2210.14215.
