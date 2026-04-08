#!/bin/bash

set -e  # Exit on error

echo "🚀 Initializing PyTorch project with uv..."

# Create directory structure
echo "📁 Creating directory structure..."
mkdir -p data models logs configs tests

# Create .gitignore
echo "📝 Creating .gitignore..."
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
*.egg-info/
dist/
build/

# Virtual Environment
.venv/
venv/
ENV/

# ML/Data
data/
*.pth
*.pt
*.ckpt
*.h5
models/*.pth
models/*.pt
checkpoints/

# Logs
logs/
*.log
runs/
wandb/

# IDE
.vscode/
.idea/
*.swp
*.swo
.DS_Store

# Jupyter
.ipynb_checkpoints/
*.ipynb

# Environment
.env
.env.local
EOF

# Create pyproject.toml
echo "📦 Creating pyproject.toml..."
cat > pyproject.toml << 'EOF'
[project]
name = "nn-project"
version = "0.1.0"
description = "PyTorch Neural Network Project"
requires-python = ">=3.9"
dependencies = [
    "torch>=2.0.0",
    "numpy>=1.24.0",
    "matplotlib>=3.7.0",
    "tensorboard>=2.13.0",
    "pyyaml>=6.0",
    "tqdm>=4.65.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "black>=23.7.0",
    "ruff>=0.0.280",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
EOF

# Create config.yaml
echo "⚙️  Creating config.yaml..."
cat > configs/config.yaml << 'EOF'
# Model Configuration
model:
  name: "MyModel"
  input_size: 784
  hidden_size: 128
  output_size: 10
  dropout: 0.2

# Training Configuration
training:
  batch_size: 64
  epochs: 10
  learning_rate: 0.001
  optimizer: "adam"
  scheduler: "step"
  weight_decay: 0.0001

# Data Configuration
data:
  train_split: 0.8
  val_split: 0.1
  test_split: 0.1
  num_workers: 4
  shuffle: true

# Logging
logging:
  log_dir: "logs"
  checkpoint_dir: "models"
  save_every: 5
  tensorboard: true
EOF

# Create utils.py
echo "🔧 Creating utils.py..."
cat > utils.py << 'EOF'
"""Utility functions for the project."""

import os
import yaml
import torch
import random
import numpy as np
from pathlib import Path


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Get the best available device (CUDA, MPS, or CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_checkpoint(model, optimizer, epoch, loss, filepath):
    """Save model checkpoint."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, filepath)
    print(f"Checkpoint saved: {filepath}")


def load_checkpoint(model, optimizer, filepath):
    """Load model checkpoint."""
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Checkpoint loaded from epoch {epoch}")
    return epoch, loss
EOF

# Create nn_0_collect_data.py
echo "📊 Creating nn_0_collect_data.py..."
cat > nn_0_collect_data.py << 'EOF'
"""
Step 0: Data Collection
Download and organize raw data for the project.
"""

import argparse
from pathlib import Path


def collect_data(data_dir: str = "data/raw"):
    """
    Collect and download data.

    Args:
        data_dir: Directory to store raw data
    """
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    print(f"Collecting data to {data_dir}...")

    # TODO: Implement data collection logic
    # Examples:
    # - Download from URL
    # - Load from API
    # - Copy from external source

    print("✓ Data collection complete!")


def main():
    parser = argparse.ArgumentParser(description="Collect data for training")
    parser.add_argument("--data-dir", type=str, default="data/raw",
                        help="Directory to store raw data")
    args = parser.parse_args()

    collect_data(args.data_dir)


if __name__ == "__main__":
    main()
EOF

# Create nn_1_prepare_data.py
echo "🔄 Creating nn_1_prepare_data.py..."
cat > nn_1_prepare_data.py << 'EOF'
"""
Step 1: Data Preparation
Preprocess data and create PyTorch datasets and dataloaders.
"""

import argparse
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
from utils import load_config, set_seed


class CustomDataset(Dataset):
    """Custom PyTorch dataset."""

    def __init__(self, data_dir: str, transform=None):
        """
        Initialize dataset.

        Args:
            data_dir: Path to data directory
            transform: Optional transforms to apply
        """
        self.data_dir = Path(data_dir)
        self.transform = transform

        # TODO: Load your data here
        self.data = []
        self.labels = []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample, label


def prepare_dataloaders(config: dict):
    """
    Prepare train, validation, and test dataloaders.

    Args:
        config: Configuration dictionary

    Returns:
        train_loader, val_loader, test_loader
    """
    # Load dataset
    dataset = CustomDataset(data_dir="data/raw")

    # Split dataset
    train_size = int(config['data']['train_split'] * len(dataset))
    val_size = int(config['data']['val_split'] * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    # Create dataloaders
    batch_size = config['training']['batch_size']
    num_workers = config['data']['num_workers']

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers
    )

    print(f"✓ Data prepared:")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")

    return train_loader, val_loader, test_loader


def main():
    parser = argparse.ArgumentParser(description="Prepare data")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                        help="Path to config file")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()

    set_seed(args.seed)
    config = load_config(args.config)

    train_loader, val_loader, test_loader = prepare_dataloaders(config)

    print(f"Batch size: {config['training']['batch_size']}")


if __name__ == "__main__":
    main()
EOF

# Create nn_2_build_model.py
echo "🏗️  Creating nn_2_build_model.py..."
cat > nn_2_build_model.py << 'EOF'
"""
Step 2: Model Building
Define neural network architecture.
"""

import argparse
import torch
import torch.nn as nn
from utils import load_config, get_device


class NeuralNetwork(nn.Module):
    """Simple feedforward neural network."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int, dropout: float = 0.2):
        """
        Initialize model.

        Args:
            input_size: Input feature size
            hidden_size: Hidden layer size
            output_size: Output size (number of classes)
            dropout: Dropout probability
        """
        super(NeuralNetwork, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """Forward pass."""
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def build_model(config: dict, device: torch.device):
    """
    Build and initialize model.

    Args:
        config: Configuration dictionary
        device: Device to place model on

    Returns:
        model: Initialized model
    """
    model_config = config['model']

    model = NeuralNetwork(
        input_size=model_config['input_size'],
        hidden_size=model_config['hidden_size'],
        output_size=model_config['output_size'],
        dropout=model_config['dropout']
    )

    model = model.to(device)

    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"✓ Model built: {model_config['name']}")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Device: {device}")

    return model


def main():
    parser = argparse.ArgumentParser(description="Build model")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                        help="Path to config file")
    args = parser.parse_args()

    config = load_config(args.config)
    device = get_device()

    model = build_model(config, device)
    print(f"\nModel architecture:\n{model}")


if __name__ == "__main__":
    main()
EOF

# Create nn_3_train.py
echo "🎯 Creating nn_3_train.py..."
cat > nn_3_train.py << 'EOF'
"""
Step 3: Model Training
Train the neural network model.
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from pathlib import Path

from nn_1_prepare_data import prepare_dataloaders
from nn_2_build_model import build_model
from utils import load_config, set_seed, get_device, save_checkpoint


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        pbar.set_postfix({'loss': loss.item(), 'acc': 100 * correct / total})

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss = running_loss / len(val_loader)
    val_acc = 100 * correct / total
    return val_loss, val_acc


def train(config: dict, seed: int = 42):
    """
    Main training function.

    Args:
        config: Configuration dictionary
        seed: Random seed
    """
    set_seed(seed)
    device = get_device()

    # Prepare data
    train_loader, val_loader, _ = prepare_dataloaders(config)

    # Build model
    model = build_model(config, device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )

    # TensorBoard
    if config['logging']['tensorboard']:
        writer = SummaryWriter(config['logging']['log_dir'])

    # Training loop
    best_val_acc = 0.0
    epochs = config['training']['epochs']

    print(f"\n🎯 Starting training for {epochs} epochs...")

    for epoch in range(1, epochs + 1):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        print(f"Epoch {epoch}/{epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # TensorBoard logging
        if config['logging']['tensorboard']:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Accuracy/train', train_acc, epoch)
            writer.add_scalar('Accuracy/val', val_acc, epoch)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
                model, optimizer, epoch, val_loss,
                f"{config['logging']['checkpoint_dir']}/best_model.pth"
            )

        # Save periodic checkpoint
        if epoch % config['logging']['save_every'] == 0:
            save_checkpoint(
                model, optimizer, epoch, val_loss,
                f"{config['logging']['checkpoint_dir']}/checkpoint_epoch_{epoch}.pth"
            )

    if config['logging']['tensorboard']:
        writer.close()

    print(f"\n✓ Training complete! Best Val Acc: {best_val_acc:.2f}%")


def main():
    parser = argparse.ArgumentParser(description="Train model")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                        help="Path to config file")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()

    config = load_config(args.config)
    train(config, args.seed)


if __name__ == "__main__":
    main()
EOF

# Create nn_4_evaluate.py
echo "📈 Creating nn_4_evaluate.py..."
cat > nn_4_evaluate.py << 'EOF'
"""
Step 4: Model Evaluation
Evaluate trained model on test set.
"""

import argparse
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

from nn_1_prepare_data import prepare_dataloaders
from nn_2_build_model import build_model
from utils import load_config, get_device


def evaluate(model, test_loader, device):
    """
    Evaluate model on test set.

    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device to run evaluation on

    Returns:
        accuracy, predictions, labels
    """
    model.eval()
    all_predictions = []
    all_labels = []
    correct = 0
    total = 0

    print("Evaluating model...")

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy, np.array(all_predictions), np.array(all_labels)


def main():
    parser = argparse.ArgumentParser(description="Evaluate model")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                        help="Path to config file")
    parser.add_argument("--checkpoint", type=str, default="models/best_model.pth",
                        help="Path to model checkpoint")
    args = parser.parse_args()

    config = load_config(args.config)
    device = get_device()

    # Prepare data
    _, _, test_loader = prepare_dataloaders(config)

    # Build model
    model = build_model(config, device)

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")

    # Evaluate
    accuracy, predictions, labels = evaluate(model, test_loader, device)

    print(f"\n📊 Test Results:")
    print(f"  Accuracy: {accuracy:.2f}%")
    print(f"\nClassification Report:")
    print(classification_report(labels, predictions))

    print(f"\nConfusion Matrix:")
    print(confusion_matrix(labels, predictions))


if __name__ == "__main__":
    main()
EOF

# Create nn_5_deploy.py
echo "🚀 Creating nn_5_deploy.py..."
cat > nn_5_deploy.py << 'EOF'
"""
Step 5: Model Deployment
Load trained model and run inference.
"""

import argparse
import torch
import numpy as np

from nn_2_build_model import build_model
from utils import load_config, get_device


class ModelInference:
    """Wrapper class for model inference."""

    def __init__(self, config_path: str, checkpoint_path: str):
        """
        Initialize inference model.

        Args:
            config_path: Path to config file
            checkpoint_path: Path to model checkpoint
        """
        self.config = load_config(config_path)
        self.device = get_device()

        # Build and load model
        self.model = build_model(self.config, self.device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        print(f"✓ Model loaded from {checkpoint_path}")
        print(f"  Device: {self.device}")

    def predict(self, input_data: np.ndarray):
        """
        Run inference on input data.

        Args:
            input_data: Input numpy array

        Returns:
            predictions: Model predictions
        """
        # Convert to tensor
        input_tensor = torch.from_numpy(input_data).float().to(self.device)

        # Run inference
        with torch.no_grad():
            outputs = self.model(input_tensor)
            _, predicted = torch.max(outputs, 1)

        return predicted.cpu().numpy()

    def predict_proba(self, input_data: np.ndarray):
        """
        Get prediction probabilities.

        Args:
            input_data: Input numpy array

        Returns:
            probabilities: Class probabilities
        """
        input_tensor = torch.from_numpy(input_data).float().to(self.device)

        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)

        return probabilities.cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description="Deploy model for inference")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                        help="Path to config file")
    parser.add_argument("--checkpoint", type=str, default="models/best_model.pth",
                        help="Path to model checkpoint")
    args = parser.parse_args()

    # Initialize inference
    inference = ModelInference(args.config, args.checkpoint)

    # Example inference
    print("\n🔮 Running example inference...")

    # TODO: Replace with actual input data
    example_input = np.random.randn(1, 784)  # Batch size 1, input size 784

    predictions = inference.predict(example_input)
    probabilities = inference.predict_proba(example_input)

    print(f"  Prediction: {predictions[0]}")
    print(f"  Probabilities: {probabilities[0]}")
    print("\n✓ Inference complete!")


if __name__ == "__main__":
    main()
EOF

# Create README.md
echo "📚 Creating README.md..."
cat > README.md << 'EOF'
# PyTorch Neural Network Project

A structured PyTorch project template initialized with `uv`.

## Setup

### 1. Initialize virtual environment

```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 2. Install dependencies

```bash
uv pip install -e .
```

For development dependencies:

```bash
uv pip install -e ".[dev]"
```

## Project Structure

```
.
├── configs/           # Configuration files
├── data/             # Data directory (gitignored)
├── logs/             # Training logs (gitignored)
├── models/           # Saved models (gitignored)
├── tests/            # Unit tests
├── nn_0_collect_data.py   # Step 0: Data collection
├── nn_1_prepare_data.py   # Step 1: Data preparation
├── nn_2_build_model.py    # Step 2: Model building
├── nn_3_train.py          # Step 3: Training
├── nn_4_evaluate.py       # Step 4: Evaluation
├── nn_5_deploy.py         # Step 5: Deployment
├── utils.py               # Utility functions
└── pyproject.toml         # Project dependencies
```

## Usage

### 1. Collect Data

```bash
python nn_0_collect_data.py --data-dir data/raw
```

### 2. Prepare Data

```bash
python nn_1_prepare_data.py --config configs/config.yaml
```

### 3. Build Model

```bash
python nn_2_build_model.py --config configs/config.yaml
```

### 4. Train Model

```bash
python nn_3_train.py --config configs/config.yaml --seed 42
```

Monitor training with TensorBoard:

```bash
tensorboard --logdir logs
```

### 5. Evaluate Model

```bash
python nn_4_evaluate.py --checkpoint models/best_model.pth
```

### 6. Deploy Model

```bash
python nn_5_deploy.py --checkpoint models/best_model.pth
```

## Configuration

Edit `configs/config.yaml` to customize:
- Model architecture
- Training hyperparameters
- Data splits
- Logging settings

## Development

Run tests:

```bash
pytest tests/
```

Format code:

```bash
black .
```

Lint code:

```bash
ruff check .
```

## License

MIT
EOF

# Create a simple test file
echo "🧪 Creating tests/test_model.py..."
cat > tests/test_model.py << 'EOF'
"""Basic tests for model."""

import torch
from nn_2_build_model import NeuralNetwork


def test_model_forward():
    """Test model forward pass."""
    model = NeuralNetwork(input_size=784, hidden_size=128, output_size=10)
    x = torch.randn(32, 784)  # Batch size 32
    output = model(x)
    assert output.shape == (32, 10)


def test_model_parameters():
    """Test model has parameters."""
    model = NeuralNetwork(input_size=784, hidden_size=128, output_size=10)
    params = list(model.parameters())
    assert len(params) > 0
EOF

# Create __init__.py for tests
touch tests/__init__.py

echo ""
echo "✅ Project initialization complete!"
echo ""
echo "Next steps:"
echo "  1. Initialize virtual environment:"
echo "     uv venv"
echo "     source .venv/bin/activate"
echo ""
echo "  2. Install dependencies:"
echo "     uv pip install -e ."
echo ""
echo "  3. Start developing:"
echo "     - Customize configs/config.yaml"
echo "     - Implement data collection in nn_0_collect_data.py"
echo "     - Run: python nn_3_train.py"
echo ""
