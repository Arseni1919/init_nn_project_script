# Script For An Initialization Of A PyTorch Project

```bash
#!/bin/bash

# List of files to create
files=(
    "nn_0_collect_data.py" 
    "nn_1_prepare_data.py" 
    "nn_2_build_model.py" 
    "nn_3_train.py" 
    "nn_4_inference.py" 
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
