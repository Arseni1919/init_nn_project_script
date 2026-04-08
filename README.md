# PyTorch Project Initialization Script

A comprehensive script to quickly set up a structured PyTorch project with `uv` package manager.

## Features

- 🐍 **UV-based setup** with `pyproject.toml`
- 📁 **Organized structure** (data/, models/, logs/, configs/, tests/)
- 🔧 **Utility functions** (config loading, seed setting, checkpointing)
- 📝 **Python boilerplate** for all workflow stages (collect, prepare, build, train, evaluate, deploy)
- ⚙️ **Config management** with YAML
- 📊 **TensorBoard integration** for training visualization
- 🧪 **Testing setup** with pytest
- 🚫 **Smart .gitignore** for ML projects

## Quick Start

Run this command in your **new project directory** to initialize everything:

```bash
bash <(curl -s https://raw.githubusercontent.com/Arseni1919/init_nn_project_script/main/init_nn_project.sh)
```

Then:

```bash
# Create and activate virtual environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv pip install -e .
```

## What Gets Created

### Directory Structure
```
your-project/
├── configs/              # Configuration files
│   └── config.yaml      # Hyperparameters and settings
├── data/                # Data directory
├── logs/                # Training logs & TensorBoard
├── models/              # Saved checkpoints
├── tests/               # Unit tests
│   ├── __init__.py
│   └── test_model.py
├── nn_0_collect_data.py    # Data collection
├── nn_1_prepare_data.py    # Dataset & DataLoader
├── nn_2_build_model.py     # Model architecture
├── nn_3_train.py           # Training loop
├── nn_4_evaluate.py        # Evaluation metrics
├── nn_5_deploy.py          # Inference
├── utils.py                # Helper functions
├── pyproject.toml          # Dependencies
├── .gitignore              # Git ignore rules
└── README.md               # Project documentation
```

### Python Files Include

Each `nn_*.py` file contains:
- ✅ Complete working boilerplate code
- ✅ Argparse for CLI arguments
- ✅ Proper imports and docstrings
- ✅ Integration with config system
- ✅ Ready to customize for your needs

### Key Features

- **utils.py**: Config loading, seed setting, device detection, checkpointing
- **config.yaml**: Centralized hyperparameters and settings
- **pyproject.toml**: Modern Python packaging with uv
- **Training**: Full training loop with validation and TensorBoard logging
- **Evaluation**: Metrics, classification report, confusion matrix
- **Deployment**: Inference wrapper class for production

## Usage Example

```bash
# 1. Collect data
python nn_0_collect_data.py --data-dir data/raw

# 2. Prepare datasets
python nn_1_prepare_data.py --config configs/config.yaml

# 3. Build model
python nn_2_build_model.py --config configs/config.yaml

# 4. Train
python nn_3_train.py --config configs/config.yaml --seed 42

# 5. Monitor training
tensorboard --logdir logs

# 6. Evaluate
python nn_4_evaluate.py --checkpoint models/best_model.pth

# 7. Deploy
python nn_5_deploy.py --checkpoint models/best_model.pth
```

## Customization

1. **Edit `configs/config.yaml`** for hyperparameters
2. **Implement `CustomDataset`** in `nn_1_prepare_data.py`
3. **Modify model architecture** in `nn_2_build_model.py`
4. **Adjust training loop** in `nn_3_train.py` as needed

## Original Script Content

```bash
#!/bin/bash

# List of files to create
files=(
    "nn_0_collect_data.py" 
    "nn_1_prepare_data.py" 
    "nn_2_build_model.py" 
    "nn_3_train.py" 
    "nn_4_evaluate.py" 
    "nn_5_deploy.py"
)

# Loop through the array and create files
for file in "${files[@]}"; do
    if [ ! -f "$file" ]; then
        touch "$file"
        echo "Created: $file"
    else
        echo "Skipped: $file (already exists)"
    fi
done
```
