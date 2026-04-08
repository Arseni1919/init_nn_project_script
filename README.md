# Script For An Initialization Of A PyTorch Project

## Quick Start

Run this command in your project directory to initialize PyTorch project files:

```bash
bash <(curl -s https://raw.githubusercontent.com/Arseni1919/init_nn_project_script/main/init_nn_project.sh)
```

## Script Content

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
