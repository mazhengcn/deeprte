# DeepRTE: neural operator for radiative transfer

[**Overview**](#overview) | [**Setup**](#setup) | [**Datasets and pretrained models**](#datasets-and-pretrained-models)

DeepRTE is a neural operator architecture designed for solving the Radiative Transfer Equation (RTE) in the phase space. This repository provides code, configuration, and utilities for training, evaluating, and experimenting with DeepRTE models using both MATLAB and Numpy datasets.


## Overview

DeepRTE learns the solution operator:

$$
  \mathcal{A}: (I_{-}; \mu_t, \mu_s, p) \to I,
$$

of the following steady-state radiative transfer equaiton,

$$
  \mathbf{\Omega} \cdot \nabla I(\mathbf{r}, \mathbf{\Omega}) + \mu_t(\mathbf{r}) I(\mathbf{r}, \mathbf{\Omega}) =
  \mu_s(\mathbf{r})\bigg(\frac{1}{\mathbb{S}_d}\int_{\mathbb{S}^{d-1}} p(\mathbf{\Omega}, \mathbf{\Omega}^*)
  I(\mathbf{r},\mathbf{\Omega}^*)\,\mathrm{d}\mathbf{\Omega}^*\bigg),
$$

with in-flow boundary condition:

$$
  I(\mathbf{r},\mathbf{\Omega}) = I_{-}(\mathbf{r},\mathbf{\Omega}), \quad\text{on } \Gamma_{-},
$$

where

$$
  \Gamma_{-} = \big\{(\mathbf{r},\mathbf{\Omega}) \mid \mathbf{n}_{\mathbf{r}}\cdot\mathbf{\Omega}<0\big\}.
$$

<!-- ## Repository Structure

```bash
deeprte/
├── .devcontainer/           # VSCode devcontainer configuration
├── data/                    # (Mounted) Datasets directory
├── src/
│   └── deeprte/
│       ├── train_lib/
│       │   └── multihost_dataloading.py  # Multi-host dataloader utilities
│       ├── config.py        # Training configuration
│       └── model/
│           └── config.py    # Model configuration
├── download_datasets.sh     # Script to download datasets
├── convert_dataset.sh       # Script to convert MATLAB to Numpy datasets
├── run_train.sh             # Script to launch training
├── run_eval.sh              # Script to launch evaluation
├── Dockerfile               # Container build file
└── README.md                # This file
``` -->

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/mazhengcn/deeprte.git --branch v1.0.0
cd deeprte
```

### 2. Install dependencies

This project uses [```jax-ai-stack```](https://github.com/jax-ml/jax-ai-stack) (JAX, Flax, Optax, Orbax, ...). The recommended way to install dependencies is using [`uv`](https://github.com/astral-sh/uv). Simply run:

```bash
uv sync
```

to install all the neccesary dependencies including the project itself or if you want to use nvidia gpu (of course you want):

```bash
uv sync --extra cuda
```

For development, run:

```bash
uv sync --dev --all-extras
```

which will install all the dev dependencies.

### 3. Container

This repository also provide a [Dockerfile](./Dockerfile) to build the runtime image for inference. You can build the image by running:

```bash
docker build -t deeprte .
```

#### Dev container

For development, this repository also provides a [devcontainer](https://code.visualstudio.com/docs/devcontainers/containers) for reproducible development. You can open the repo root folder in VSCode and it will automatically build the container with all dependencies and developing tools, and also the needed data folder volume mount. The python and its dependencies are also managed by `uv`.

The devcontainer [.devcontainer/Dockerfile](./.devcontainer/Dockerfile) and [.devcontainer/devcontainer.json](./.devcontainer/devcontainer.json) config can be found under [.devcontainer](./.devcontainer) folder. You can modify it for your own preference.

## Datasets and pretrained models

### Download datasets

The datasets for train and test deeprte are generated using conventional numerical methods writtern in MATLAB and Python. The source code can be found in a separate repo [rte-dataset](https://github.com/mazhengcn/rte-dataset), for more detailed information, please check that repo.

The datasets for inference (test) and pretrain are in Huggingface repo https://huggingface.co/datasets/mazhengcn/rte-dataset. The datasets can be download to `DATA_DIR` by (the huggingface-cli should be installed first, if you follow above setup then you already have it, otherwise check https://huggingface.co/docs/huggingface_hub/guides/cli):

```bash
huggingface-cli download mazhengcn/rte-dataset \
    --exclude=interim/* \
    --repo-type=dataset \
    --local-dir=${DATA_DIR}
```

The resulting folder structure should be (for inference we only need datasets under `raw/test`):

```bash
${DATA_DIR}
├── processed
│   └── tfds      # Processed TFDS dataset for pretraining.
├── raw
│   ├── test      # Raw Matlab dataset for test/inference.
│   └── train     # Raw Matlab dataset for pretraining using grain.
└── README.md
```

Each Matlab dataset contains the following keys:

| Key            | Array Shape     | Description                                 |
| -------------- | --------------- | ------------------------------------------- |
| `list_Psi`     | `[2M, I, J, N]` | Numerical solutions (labels)                |
| `list_psiL`    | `[M, J, N]`     | Left boundary values                        |
| `list_psiR`    | `[M, J, N]`     | Right boundary values                       |
| `list_psiB`    | `[M, I, N]`     | Bottom boundary values                      |
| `list_psiT`    | `[M, I, N]`     | Top boundary values                         |
| `list_sigma_a` | `[I, J, N]`     | Absorption coefficient                      |
| `list_sigma_T` | `[I, J, N]`     | Total coefficient                           |
| `ct`, `st`     | `[1, M]`        | Discrete velocity coordinates (quadratures) |
| `omega`        | `[1, M]`        | Weights of velocity coordinates             |

### Download pretrained models

The pretrained models can be loaded to `MODEL_DIR` from Huggingface using:

```bash
huggingface-cli download mazhengcn/deeprte \
    --repo-type=model \
    --local-dir=${MODELS_DIR}
```

with the following folder structure:

```bash
${MODELS_DIR}
├── README.md
├── v0                    # Pre-release version corresponding to branch deeprte-haiku, depracated.
└── v1                    # Current release models for difference scattering kernel range.
    ├── g0.1
    │   ├── config.json
    │   └── params
    ├── g0.5
    │   ├── config.json
    │   └── params
    └── g0.8
        ├── config.json
        └── params
```

We provide a convient shell script at [scripts/download_dataset_and_models.sh](./scripts/download_dataset_and_models.sh) to download datasets to [./data](./data/) and pretrained models under [./models](./models/) by

```bash
uv run ./scrips/download_dataset_and_models.sh
```

## Run DeepRTE



## Training

To start a training experiment:

```bash
./run_train.sh <DATA_PATH>
```

- Reads training config from `deeprte/config.py` and model config from `deeprte/model/config.py`.
- Loads dataset from `<DATA_PATH>`.
- Runs training and evaluation in multithreaded mode.
- Saves configs, checkpoints, and model parameters to `./data/experiments/square_full_*_${TIMESTAMP}`.

---

## Evaluation

To evaluate a trained model:

```bash
./run_eval.sh <RESTORE_PATH> <TEST_DATA_PATH> <EVAL_CKPT_DIR>
```

- Reads configs from `deeprte/config.py` and `deeprte/model/config.py`.
- Loads model parameters from `<RESTORE_PATH>`.
- Loads evaluation dataset from `<TEST_DATA_PATH>`.
- Saves evaluation logs to `<EVAL_CKPT_DIR>`.

---

## Development Environment

This repository is ready for use with VSCode Dev Containers. The `.devcontainer/devcontainer.json` configures:

- Python, Jupyter, TensorBoard, and other useful extensions
- GPU access (`--gpus=all`)
- Shared memory and security options
- Volume mounts for data, virtual environments, and cache
- Automatic environment sync and pre-commit hooks

To use, simply open the folder in VSCode and select "Reopen in Container".

---

## References

- [Radiative Transfer Equation (Wikipedia)](https://en.wikipedia.org/wiki/Radiative_transfer_equation)
- [JAX Documentation](https://jax.readthedocs.io/)
- [VSCode Dev Containers](https://code.visualstudio.com/docs/devcontainers/containers)

---

For questions or contributions, please open an issue or pull request.
