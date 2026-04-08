# PyTorch Project Initialization Script

Minimal PyTorch project setup with CIFAR-10, ready to run.

## Features

- 🚀 **Minimal code** - bare essentials only
- 🎯 **CIFAR-10 ready** - works out of the box
- 📊 **W&B integration** - experiment tracking
- ⚙️ **Python config** - simple dict-based config
- 🔧 **UV package manager** - fast dependency management

## Quick Start

```bash
bash <(curl -s https://raw.githubusercontent.com/Arseni1919/init_nn_project_script/main/init_nn_project.sh)
```

Then:

```bash
uv venv && source .venv/bin/activate
uv pip install -e .
```

Add your W&B key to `.env`:
```bash
WANDB_API_KEY=your_key_here
```

Run:
```bash
python nn_0_collect_data.py
python nn_3_train.py
python nn_4_evaluate.py
```

## What Gets Created

```
your-project/
├── data/                   # CIFAR-10 dataset
├── models/                 # Saved models
├── logs/                   # Logs
├── nn_0_collect_data.py   # Download CIFAR-10
├── nn_1_prepare_data.py   # DataLoaders
├── nn_2_build_model.py    # Simple CNN
├── nn_3_train.py          # Training with W&B
├── nn_4_evaluate.py       # Test evaluation
├── nn_5_deploy.py         # ONNX export & inference
├── config.py              # Hyperparameters
├── utils.py               # Helpers
├── pyproject.toml         # Dependencies
├── .env                   # W&B API key
└── .gitignore
```

## Philosophy

- **No argparse** - edit config.py instead
- **No comments** - code is self-explanatory
- **No empty lines** in functions
- **Minimal dependencies** - torch, torchvision, wandb, dotenv
- **Working example** - CIFAR-10 runs immediately
