# DeepRTE

## Prepare datasets

### Download datasets

First we need to copy datasets to our `/path/to/project_root` folder and currently the downloaded datasets are MATLAB `.mat` files. Run the following command:

- Default:

  ```bash
  ./download_datasets.sh
  ```

  will download all datasets using `rsync` from directory `/cluster/home/xuzhiqin_02/rte_data/` in host `xuzhiqin_02@202.120.13.117` to `.data/matlab/` (if this dir exists, otherwise you need to create it).

- With custom arguments:

  ```bash
  ./download_datasets.sh <SERVER_URL> <REMOTE_DATA_DIR> <DOWNLOAD_DIR>
  ```

  will download datasets from `SERVER_URL:REMOTE_DATA_DIR` to `DOWNLOAD_DIR`.

### MATLAB Datasets

After previous step, you should have your MATLAB datasets in directory `<DOWNLOAD_DIR>`. By defaults, you should see the following datasets:

```bash
e1_L_delta_*.mat, e1_R_delta_*.mat, e1_B_delta_*.mat, e1_T_delta_*.mat
```

which stands for 4 boundary positions (left, right, bottom, top) and other datasets. For each MATLAB dataset, it has the following keys and array shapes

| Key            | Array Shape     | Description                                          |
| -------------- | --------------- | ---------------------------------------------------- |
| `list_Psi`     | `[2M, I, J, N]` | numerical solutions as training labels               |
| `list_psiL`    | `[M, J, N]`     | left boundary values                                 |
| `list_psiR`    | `[M, J, N]`     | right boundary values                                |
| `list_psiB`    | `[M, I, N]`     | bottom boundary values                               |
| `list_psiT`    | `[M, I, N]`     | top boundary values                                  |
| `list_sigma_a` | `[I, J, N]`     | absorption coefficient function                      |
| `list_sigma_T` | `[I, J, N]`     | total coefficient function                           |
| `ct` and `st`  | `[1, M]`        | discrete coordinates (quadratures) in velocity space |
| `omega`        | `[1, M]`        | weights of velocity coordinates                      |

### Convert datasets

Then we need to convert/concatenate MATLAB datasets into one single `*.npz` dataset for training and testing:

- Default:

  ```bash
  ./convert_dataset.sh
  ```

  which will convert

  ```bash
  e1_L_delta_*.mat, e1_R_delta_*.mat, e1_B_delta_*.mat, e1_T_delta_*.mat
  ```

  into `./data/train/square_full_*.npz`.

- With custom arguments:

  ```bash
  ./convert_dataset.sh <SOURCE_DIR> <DATAFILES> <SAVE_PATH>
  ```

  which will converte a list of datasets with names `DATAFILES` under directory `SOURCE_DIR` to `SAVE_PATH`.

**Note:** all the arguments have default values, please check [`conver_dataset.sh`](./convert_dataset.sh) for details.

### Numpy dataset

The Numpy dataset will be used for training, testing and evaluating the DeepRTE model. Besides obtaining the dataset from MATLAB code, you can also provide it using any method as long as it is saved as the following flat Numpy dict:

| Key              | Array shape     | Description                                                   |
| ---------------- | --------------- | ------------------------------------------------------------- |
| `data/psi_label` | `[N, I, J, 2M]` | label solutions                                               |
| `data/psi_bc`    | `[N, I*J, M]`   | boundary values                                               |
| `data/sigma_t`   | `[N, I, J]`     | total coefficient values on grid                              |
| `data/sigma_a`   | `[N, I, J]`     | absorption values on grid                                     |
| `data/phi`       | `[N, I, J]`     | density for evaluation only                                   |
| `grid/r`         | `[I, J, d]`     | position coordinates on the grid                              |
| `grid/v`         | `[2M, d]`       | velocity coordinates (quadratures)                            |
| `grid/w_angle`   | `[2M]`          | weights (quadratures) associated with velocity coordinates    |
| `grid/rv_prime`  | `[I*J, M, 2d]`  | phase space coordinates by concat of `r` and `v`              |
| `grid/w_prime`   | `[I*J, M]`      | weights (quadratures) associated with phase space coordinates |

**Note:** during training the flat numpy dict is loaded and converted to a nest dict consisting of `"data"` and `"grid"` as two subdicts and then be processed separately. Here is an example:

```bash
{
    'data': {'sigma_t': (2000, 40, 40), 'sigma_a': (2000, 40, 40), 'psi_label': (2000, 40, 40, 24), 'phi': (2000, 40, 40), 'psi_bc': (2000, 160, 12)},
    'grid': {'r': (40, 40, 2), 'v': (24, 2), 'w_angle': (24,), 'rv_prime': (160, 12, 4), 'w_prime': (160, 12)}
}
```

## Run DeepRTE training experiment

The training task can be excuted by a simple command:

```bash
./run_train.sh <DATA_PATH>
```

which will do the following things:

- Read train configuration from [`deeprte/config.py`](./deeprte/config.py) and model configuration from [`deeprte/model/config.py`](./deeprte/model/config.py).
- Load training dataset from `DATA_PATH`.
- Run training and evaluation in multithread mode.
- Save configs, checkpoints, model parameters, etc. into [`./data/experiments`](./data/experiments/)`/square_full_*_${TIMESTAMPS}`.

## Run DeepRTE evaluation

The evaluation can be processed by the following command:

```bash
./run_eval.sh <RESTORE_PATH> <TEST_DATA_PATH> <EVAL_CKPT_DIR>
```

which will do:

- Read configs from [`deeprte/config.py`](./deeprte/config.py) and [`deeprte/model/config.py`](./deeprte/model/config.py).
- Load model parameters from `RESTORE_PATH`.
- Load evaluation dataset from `TEST_DATA_PATH`.
- Save evaluation logs to `EVAL_CKPT_DIR`.

**Note:** all the arguments have default values, please check [`run_eval.sh`](run_eval.sh) for reference.
