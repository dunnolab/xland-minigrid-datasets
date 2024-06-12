# Dataset collection

Here we provide the code used to collect the datasets. We adapted the single-task recurrent PPO implementation from the original XLand-MiniGrid baselines. We used wandb sweeps to pretraind base agent and collect individual learning histories at scale on multiple GPUs. We then combined all the individual histories into a single dataset using `combine.py`.

If you notice any discrepancies with the paper, don't be afraid to open an issue and report about it! 

## Pretraining

Pretraining is simple. We provide config for pretraining in `configs/pretrain_base.yaml`. To start:
```commandline
python training/train.py \
    --config_path='configs/pretrain_base.yaml' \
    --checkpoint_path='path-for-the-final-checkpoint' \
    --wandb_logging=True
```
We used pretraining only for the main dataset (tasks from medium benchmark).

## Collecting

We used wandb sweeps for collection. We provide base configs for trivial and medium in `configs/trivial_base.yaml` and `configs/medium_base.yaml` respectively.

### Trivial

First, create wandb config:
```yaml
# trivial_wandb.yaml
entity: <your-enitty>
project: xminigrid-datasets
program: training/train.py
method: grid
parameters:
  config_path:
    value: "configs/trivial_base.yaml"
  group:
    value: "xland-minigrid-datasets-trivial-v0"
  dataset_path:
    value: <path-to-your-dir-for-data>
  dataset_num_histories:
    value: 32
  ruleset_id:
    min: 0
    max: 10000
    distribution: int_uniform
```
Next, create wandb agent with the `wandb sweep trivial_wandb.yaml` to get the sweep ID. To start collection, run `wandb agent <sweep-id>`. 

### Medium

Likewise, create a config:
```yaml
# medium_wandb.yaml
entity: <your-enitty>
project: xminigrid-datasets
program: training/train.py
method: grid
parameters:
  config_path:
    value: "configs/medium_base.yaml"
  group:
    value: "xland-minigrid-datasets-medium-v0"
  dataset_path:
    value: <path-to-your-dir-for-data>
  pretrained_checkpoint_path:
    value: <path-to-your-pre-trained-checkpoint>
  dataset_num_histories:
    value: 32
  ruleset_id:
    min: 0
    max: 30000
    distribution: int_uniform
```
Unlike trivial, you must additionally specify the path to the pre-trained checkpoint (you can use `None` to train from scratch). After that, create wandb agent with the `wandb sweep medium_wandb.yaml` to get the sweep ID. To start collection, run `wandb agent <sweep-id>`.

## Combining

We used simple `combine.py` script to combine all individual learning histories into one dataset. As we described in the paper, we already tuned the hdf5 chunk size which worked best in our experiments, however you can customise it by changing the hardcoded values in the code.

For example, we filterd out all runs with last return below 0.3:
```commandline
python combine.py \
    --wandb-entity=your-entity \
    --wandb-sweep=your-collection-sweep \
    --data-path=your-data-path \
    --combined-path=your-combined-path \
    --final-return-thrs=0.3 \
```
