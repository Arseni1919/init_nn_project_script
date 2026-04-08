#!/bin/bash

set -e

echo "🚀 Initializing PyTorch project with uv..."

echo "📁 Creating directory structure..."
mkdir -p data models logs

echo "📝 Creating .gitignore..."
cat > .gitignore << 'EOF'
__pycache__/
*.py[cod]
.venv/
data/
*.pth
*.pt
models/
logs/
wandb/
.DS_Store
.env
EOF

echo "📦 Creating pyproject.toml..."
cat > pyproject.toml << 'EOF'
[project]
name = "nn-project"
version = "0.1.0"
requires-python = ">=3.9"
dependencies = [
    "torch>=2.0.0",
    "torchvision>=0.15.0",
    "numpy>=1.24.0",
    "wandb>=0.15.0",
    "python-dotenv>=1.0.0",
    "onnx>=1.14.0",
    "onnxruntime>=1.15.0",
]
EOF

echo "⚙️  Creating config.py..."
cat > config.py << 'EOF'
config = {
    "batch_size": 64,
    "epochs": 10,
    "lr": 0.001,
    "device": "auto",
    "seed": 42,
}
EOF

echo "🔧 Creating utils.py..."
cat > utils.py << 'EOF'
import torch
import random
import numpy as np

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
EOF

echo "📊 Creating nn_0_collect_data.py..."
cat > nn_0_collect_data.py << 'EOF'
import torchvision
import torchvision.transforms as transforms
from pathlib import Path

def main():
    Path("data").mkdir(exist_ok=True)
    transform = transforms.Compose([transforms.ToTensor()])
    torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    print("✓ CIFAR-10 downloaded")
    print(f"Train size: 50000")
    print(f"Test size: 10000")

if __name__ == "__main__":
    main()
EOF

echo "🔄 Creating nn_1_prepare_data.py..."
cat > nn_1_prepare_data.py << 'EOF'
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from config import config

class CIFAR10Dataset(Dataset):
    def __init__(self, train=True):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.dataset = torchvision.datasets.CIFAR10(root='./data', train=train, download=False)
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        image = self.transform(image)
        return image, label

def get_dataloaders():
    train_dataset = CIFAR10Dataset(train=True)
    test_dataset = CIFAR10Dataset(train=False)
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)
    return train_loader, test_loader

def main():
    train_loader, test_loader = get_dataloaders()
    images, labels = next(iter(train_loader))
    print(f"✓ Dataset and DataLoaders ready")
    print(f"Train size: {len(train_loader.dataset)}")
    print(f"Test size: {len(test_loader.dataset)}")
    print(f"Batch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")

if __name__ == "__main__":
    main()
EOF

echo "🏗️  Creating nn_2_build_model.py..."
cat > nn_2_build_model.py << 'EOF'
import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def get_model():
    return SimpleNet()

def main():
    model = get_model()
    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    print(f"✓ Model created")
    print(f"Input: {x.shape}")
    print(f"Output: {out.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

if __name__ == "__main__":
    main()
EOF

echo "🎯 Creating nn_3_train.py..."
cat > nn_3_train.py << 'EOF'
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from pathlib import Path
from dotenv import load_dotenv
from nn_1_prepare_data import get_dataloaders
from nn_2_build_model import get_model
from utils import set_seed, get_device
from config import config

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    train_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    return train_loss / len(train_loader)

def train():
    load_dotenv()
    set_seed(config["seed"])
    device = get_device()
    wandb.init(project="nn-project", config=config)
    train_loader, test_loader = get_dataloaders()
    model = get_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    Path("models").mkdir(exist_ok=True)
    for epoch in range(config["epochs"]):
        avg_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        wandb.log({"epoch": epoch, "loss": avg_loss})
        print(f"Epoch {epoch+1}/{config['epochs']}, Loss: {avg_loss:.4f}")
    torch.save(model.state_dict(), "models/model.pth")
    wandb.finish()
    print("✓ Training complete")

def main():
    x = torch.randn(2, 3, 32, 32)
    model = get_model()
    out = model(x)
    print(f"✓ Training script ready")
    print(f"Test forward pass: {out.shape}")

if __name__ == "__main__":
    main()
EOF

echo "📈 Creating nn_4_evaluate.py..."
cat > nn_4_evaluate.py << 'EOF'
import torch
from nn_1_prepare_data import get_dataloaders
from nn_2_build_model import get_model
from utils import get_device

def evaluate():
    device = get_device()
    _, test_loader = get_dataloaders()
    model = get_model().to(device)
    model.load_state_dict(torch.load("models/model.pth", map_location=device))
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f"✓ Test Accuracy: {accuracy:.2f}%")
    return accuracy

def main():
    device = get_device()
    model = get_model().to(device)
    x = torch.randn(2, 3, 32, 32).to(device)
    model.eval()
    with torch.no_grad():
        out = model(x)
    print(f"✓ Evaluation script ready")
    print(f"Test inference: {out.shape}")

if __name__ == "__main__":
    main()
EOF

echo "🚀 Creating nn_5_deploy.py..."
cat > nn_5_deploy.py << 'EOF'
import torch
import numpy as np
import onnx
import onnxruntime as ort
from pathlib import Path
from nn_2_build_model import get_model
from utils import get_device

def export_to_onnx():
    device = get_device()
    model = get_model().to(device)
    model.load_state_dict(torch.load("models/model.pth", map_location=device))
    model.eval()
    Path("models").mkdir(exist_ok=True)
    dummy_input = torch.randn(1, 3, 32, 32).to(device)
    torch.onnx.export(model, dummy_input, "models/model.onnx",
                      input_names=['input'], output_names=['output'],
                      dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
    onnx_model = onnx.load("models/model.onnx")
    onnx.checker.check_model(onnx_model)
    print("✓ Model exported to ONNX")
    print(f"Saved: models/model.onnx")

def predict(image):
    ort_session = ort.InferenceSession("models/model.onnx")
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    outputs = ort_session.run(None, {'input': image})
    pred = np.argmax(outputs[0], axis=1)
    return pred

def main():
    x = torch.randn(1, 3, 32, 32)
    device = get_device()
    model = get_model().to(device)
    model.eval()
    with torch.no_grad():
        out = model(x.to(device))
    print(f"✓ Deployment script ready")
    print(f"Input: {x.shape}")
    print(f"Output: {out.shape}")

if __name__ == "__main__":
    main()
EOF

echo "🔑 Creating .env..."
cat > .env << 'EOF'
WANDB_API_KEY=your_wandb_api_key_here
EOF

echo "📚 Creating README.md..."
cat > README.md << 'EOF'
# PyTorch CIFAR-10 Project

Minimal PyTorch project for CIFAR-10 classification.

## Setup

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

Add your W&B API key to `.env`:
```
WANDB_API_KEY=your_key_here
```

## Usage

```bash
python nn_0_collect_data.py
python nn_1_prepare_data.py
python nn_2_build_model.py
python nn_3_train.py
python nn_4_evaluate.py
python nn_5_deploy.py
```

## Configuration

Edit `config.py` to change hyperparameters.
EOF

echo ""
echo "✅ Project initialized!"
echo ""
echo "Next steps:"
echo "  1. uv venv && source .venv/bin/activate"
echo "  2. uv pip install -e ."
echo "  3. Add WANDB_API_KEY to .env"
echo "  4. python nn_0_collect_data.py"
echo "  5. python nn_3_train.py"
echo ""
