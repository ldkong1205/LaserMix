<img src="../docs/figs/logo.png" align="right" width="20%">

# Getting Started
- [Train Predefined Models on Standard Datasets](#train-predefined-models-on-standard-datasets)
  - [Train with a single GPU](#hourglass-train-with-a-single-gpu)
  - [Train with multiple GPUs](#hourglass-train-with-multiple-gpus)
  - [Train with multiple machines](#hourglass-train-with-multiple-machines)
- [Test Existing Models on Standard Datasets](#test-existing-models-on-standard-datasets)


## Train Predefined Models on Standard Datasets

- This codebase implements distributed training and non-distributed training, which uses `MMDistributedDataParallel` and `MMDataParallel`, respectively.

- All outputs (log files and checkpoints) will be saved to the working directory, which is specified by `work_dir` in the config file.

- :warning: **Important:** By default, we evaluate the model on the validation set after each epoch, **without** using extra tricks including model ensemble and test-time-augmentation (TTA). To ensure fair comparisons, we advise you to follow the **same** configuration strictly.

- You can change the evaluation interval by adding the interval argument in the training config.
  ```Shell
  train_cfg = dict(type='EpochBasedTrainLoop', val_interval=1)
  ```
- **Note:** The default learning rate in config files is for 8 GPUs and the exact batch size is marked by the config’s file name, e.g., `‘2xb8’` means 2 samples per GPU using 8 GPUs. According to the Linear Scaling Rule, you might need to set the learning rate proportional to the batch size if you use different GPUs or images per GPU, e.g., `lr=0.01` for 4 GPUs * 2 img/gpu and `lr=0.08` for 16 GPUs * 4 img/gpu. However, since most of the models in this repo use Adam rather than SGD for optimization, the rule may not hold and users need to tune the learning rate by themselves.

### :hourglass: Train with a single GPU
- The default command is as follows:
  ```python
  python tools/train.py ${CONFIG_FILE} [optional arguments]
  ```

- :memo: For example, to train `LaserMix` with the `Cylinder3D` backbone on `SemanticKITTI` under a `10%` annotation budget using a single GPU, run the following command:
  ```python
  python tools/train.py configs/lasermix/lasermix_cy3d_semi_semantickitti_10.py
  ```
- **Note:** If you want to specify the working directory in the command, you can add an argument `--work-dir ${YOUR_WORK_DIR}`.


### :hourglass: Train with multiple GPUs
- The default command is as follows:
  ```python
  ./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} [optional arguments]
  ```
- :memo: For example, to train `LaserMix` with the `Cylinder3D` backbone on `SemanticKITTI` under a `10%` annotation budget using four GPUs, run the following command:
  ```python
  ./tools/dist_train.sh configs/lasermix/lasermix_cy3d_semi_semantickitti_10.py 4
  ```
- **Note:** Optional argument: `--cfg-options 'Key=value'`, which overrides some settings in the used config.


### :hourglass: Train with multiple machines
- If you run MMDetection3D on a cluster managed with `slurm`, you can use the script `slurm_train.sh`. This script also supports single-machine training.
  ```python
  [GPUS=${GPUS}] ./tools/slurm_train.sh ${PARTITION} ${JOB_NAME} ${CONFIG_FILE} ${WORK_DIR}
  ```

- :memo: For example, to train `LaserMix` with the `Cylinder3D` backbone on `SemanticKITTI` under a `10%` annotation budget using `slurm`, run the following command:
  ```python
  GPUS=4 ./tools/slurm_train.sh partition lasermix configs/lasermix/lasermix_cy3d_semi_semantickitti_10.py /logs/semi/semantickitti/lasermix_cy3d_10
  ```

## Test Existing Models on Standard Datasets
Coming soon.
