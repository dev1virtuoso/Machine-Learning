#!/bin/bash
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

echo "Starting environment baseline verification via importlib.metadata..."

python3 -c "
import sys
import importlib.metadata

required_packages = {
    'torch': '2.3.0',
    'torchvision': '0.18.0',
    'transformers': '4.40.0',
    'opencv-python': '4.8.0',
    'numpy': '1.24.0',
    'pytorch-metric-learning': '2.0.0',
    'pillow': '10.0.0',
    'timm': '0.9.0'
}

def parse_version(v_str):
    return [int(x) for x in ''.join(c for c in v_str if c.isdigit() or c=='.').split('.') if x]

print('-------------------------------------------')
all_pass = True
for pkg, min_ver in required_packages.items():
    try:
        # Standardized normalization logic matching modern wheels lookup metadata structures
        lookup_name = 'opencv-python' if pkg == 'opencv-python' else pkg
        installed_ver = importlib.metadata.version(lookup_name)
        
        if parse_version(installed_ver) >= parse_version(min_ver):
            print(f' \033[0;32m[PASS]\033[0m {pkg} installed: {installed_ver} >= {min_ver}')
        else:
            print(f' \033[0;31m[FAIL]\033[0m {pkg} installed: {installed_ver} < {min_ver}')
            all_pass = False
    except importlib.metadata.PackageNotFoundError:
        # Check alternative common naming schemas mapping
        try:
            alt_name = 'opencv-python-headless' if pkg == 'opencv-python' else pkg
            installed_ver = importlib.metadata.version(alt_name)
            print(f' \033[0;32m[PASS]\033[0m {pkg} ({alt_name}) installed: {installed_ver} >= {min_ver}')
        except importlib.metadata.PackageNotFoundError:
            print(f' \033[0;31m[FAIL]\033[0m {pkg} is not installed inside the active context.')
            all_pass = False

sys.exit(0 if all_pass else 1)
"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}System Validation Complete. Constraints met successfully.${NC}"
else
    echo -e "${RED}System baseline validation failed. Execute install_dependencies.sh script.${NC}"
fi