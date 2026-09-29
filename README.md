# Geometric Laplace Neural Operator (GLNO)

Research code for learning operators on irregular geometric domains and regular grids. GLNO uses geometry-aware features and Laplace-domain pole-residue layers to model mappings between fields defined on meshes or grids.

The repository also contains dataset loaders, preprocessing utilities, baseline model implementations, training scripts, and visualization helpers for several scientific machine-learning tasks.

## Repository Status

The primary implementation is [model/GLNO/layers.py](model/GLNO/layers.py). The model registry currently exposes:

- `GLNO`: mesh and irregular-domain operator
- `GLNO1D`: one-dimensional grid operator
- `GLNO2D`: two-dimensional grid operator

To run the experiment, please prepare data in `data/` directory and setup training config in `config/`d directory.

## Supported Tasks

The dataset registry in [dataset/__init__.py](dataset/__init__.py) currently contains the following task names:

| Task | Dataset type | Expected storage |
| --- | --- | --- |
| `poisson` | Mesh / PDE | `data/poisson/{train,val,test}.h5` |
| `car` | Mesh CFD | `data/car/{train,val,test}.h5` |
| `cylinder_flow` | Mesh CFD | `data/cylinder_flow/{train,val,test}.h5` |
| `rna` | Molecular mesh | `data/rna` |
| `human` | Human-shape mesh | `data/human` |
| `shrec11_simplified` | Shape mesh | task-specific mesh data |
| `pendulum`, `lorenz`, `duffing` | 1D grid data | `data/<task>/<case>/train.pt` |
| `beam`, `diffusion`, `reacdiffusion` | 2D grid data | `data/<task>/{train,vali,test}.pt` |
| `turbulent` | 2D turbulent-flow data | `data/turbulent/{train,val,test}.pt` |

The paths above describe the conventions used by the loaders. Check the corresponding classes in [dataset/glno_dataset__.py](dataset/glno_dataset__.py) and [dataset/grid_dataset.py](dataset/grid_dataset.py) before adding a new dataset.

## Installation

Create an environment with a PyTorch build compatible with your CUDA version, then install the Python dependencies used by the repository:

```bash
conda create -n glno python=3.10
conda activate glno

# Install a CUDA-compatible PyTorch build from pytorch.org first.
pip install numpy scipy pyyaml tqdm h5py einops potpourri3d torch-geometric
```

The correct installation command for `torch-geometric` depends on the installed PyTorch and CUDA versions. Follow the official PyTorch Geometric installation instructions if the simple command above is not sufficient.

Run commands from the repository root:

```bash
cd path/to/GLNO
```

## Configuration

`train.py` loads a task-specific YAML file using this pattern:

```text
config/<task-prefix>/<config-name>
```

For example, the command below expects:

```text
config/poisson/config_glno.yaml
```

The configuration must define at least the `system`, `dataset`, `model`, and `training` sections. Use a local `config_glno.yaml` template if one is available in your checkout, or use the configuration files from the experiment that produced your dataset and checkpoint.

Important model settings include `C_width`, `k_eig`, `glno_sigma`, `glno_poles`, normalization options, and the input/output channel definitions. The dataset section controls batch sizes, validation splitting, and padding for variable-size meshes.

## Training

The main entry point is [train.py](train.py). A minimal single-GPU command is:

```bash
python train.py --task=poisson --local_rank=0 --config=config_glno.yaml
```

For a grid task:

```bash
python train.py --task=lorenz/rho10 --local_rank=0 --config=config_glno.yaml
```

Available command-line options include:

| Option | Description |
| --- | --- |
| `--task` | Required task name, such as `poisson`, `car`, or `lorenz/rho10`. |
| `--config` | YAML filename selected under `config/<task-prefix>/`; default: `config_glno.yaml`. |
| `--local_rank` | CUDA device index for single-process execution. |
| `--distributed` | Enables distributed initialization when launched with `torchrun`. |
| `--dataset_name` | Processed HDF5 filename for `cortex`, `intra`, and `poissonunstruc`. |
| `--load_model` | Loads a model state dictionary before training or evaluation. |
| `--evaluate` | Skips training and evaluates the loaded model on the test set. |
| `--seed` | Random seed; default: `42`. |
| `--channels` | Overrides the configured model width. |
| `--modes` | Overrides the configured spectral/Fourier mode count. |
| `--sigma` | Overrides the configured sigma count. |
| `--number_worker` | DataLoader worker count; default: `4`. |

## Evaluation

To evaluate a saved model without running another training phase, provide both `--evaluate` and `--load_model`:

```bash
python train.py \
  --task=poisson \
  --local_rank=0 \
  --evaluate \
  --load_model=logs/poisson/09_28/GLNO_12_00_00/best_model.pth
```

Each training run creates a directory under:

```text
logs/<task>/<MM_DD>/<model>_<HH_MM_SS>/
```

Depending on the configuration, the directory may contain:

- `config.yaml`: resolved training configuration and command arguments
- `train.log`: console and training log output
- `best_model.pth`: best validation checkpoint saved by the training loop
- `checkpoint_<epoch>_<loss>.pth`: optional full training checkpoints
- `test_loss_distribution.png`: test-loss distribution when enabled

## Data Preparation

The repository includes preparation utilities for several data formats:

- [dataset/prepare_data.py](dataset/prepare_data.py): mesh/PDE preprocessing helpers
- [dataset/prepare_grid_dataset.py](dataset/prepare_grid_dataset.py): grid dataset conversion
- [dataset/prepare_cortex.py](dataset/prepare_cortex.py): cortex preprocessing
- [process_data.py](process_data.py): Laplacian and geometric operator preprocessing

Mesh HDF5 files are expected to contain sample groups and metadata required by the mesh loaders. Depending on the task, these fields include `input`, `output`, `vertices`, `mass`, `eval`, `evec`, and geometric features such as `distance`. Grid `.pt` files are expected to contain tensors named `x`, `y`, and grid coordinates such as `grid_x` and `grid_y`.

For variable-size mesh batches, the training pipeline uses task-specific collate functions from [utils/Dataloader_Training.py](utils/Dataloader_Training.py). Keep the loader, collate function, model input contract, and YAML configuration consistent when adding a new dataset.

## Adding a Dataset or Model

1. Implement the dataset class in `dataset/`.
2. Register it in `DATASET_DICT` in [dataset/__init__.py](dataset/__init__.py).
3. Add any task-specific collate function to [utils/Dataloader_Training.py](utils/Dataloader_Training.py).
4. Add the matching YAML configuration under `config/<task-prefix>/`.
5. Implement the model in `model/` and register it in `MODEL_DICT` in [model/__init__.py](model/__init__.py).
6. Add or update a task visualizer in [utils/visualizor](utils/visualizor) when rendered predictions are needed.
7. Run a small training/evaluation pass before launching a full experiment.

## Project Layout

```text
GLNO/
├── dataset/       Dataset classes and preprocessing scripts
├── model/         GLNO, grid GLNO, and baseline architectures
├── utils/         Losses, dataloaders, geometry, logging, and visualization
├── train.py       Main configurable training and evaluation entry point
└── README.md      Project documentation
```