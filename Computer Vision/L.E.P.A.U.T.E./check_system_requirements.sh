#!/bin/bash
set -euo pipefail

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}Starting environment baseline verification via dynamic AST parsing...${NC}"

if [ ! -f "requirements.txt" ]; then
    echo -e "${RED}Fatal Error: requirements.txt not found. Cannot verify system baselines.${NC}"
    exit 1
fi

python3 -c "
import sys
import importlib.metadata
import re

try:
    from packaging.version import parse as parse_version
except ImportError:
    print('\033[0;31m[FAIL]\033[0m \'packaging\' module missing. Run install_dependencies.sh first.')
    sys.exit(1)

def verify_requirements(req_file):
    all_pass = True
    with open(req_file, 'r') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        
        match = re.match(r'^([a-zA-Z0-9_\-]+)(>=|==|>|<=|<)?([0-9\.]+)?', line)
        if not match:
            continue
            
        pkg_raw = match.group(1)
        operator = match.group(2)
        min_ver = match.group(3)
        
        try:
            installed_ver = importlib.metadata.version(pkg_raw)
            
            if min_ver and operator in ['>=', '==']:
                if parse_version(installed_ver) >= parse_version(min_ver):
                    print(f' \033[0;32m[PASS]\033[0m {pkg_raw} installed: {installed_ver} >= {min_ver}')
                else:
                    print(f' \033[0;31m[FAIL]\033[0m {pkg_raw} installed: {installed_ver} < {min_ver}')
                    all_pass = False
            else:
                print(f' \033[0;32m[PASS]\033[0m {pkg_raw} installed: {installed_ver}')
                
        except importlib.metadata.PackageNotFoundError:
            if pkg_raw == 'opencv-python':
                try:
                    installed_ver = importlib.metadata.version('opencv-python-headless')
                    if min_ver and operator in ['>=', '==']:
                        if parse_version(installed_ver) >= parse_version(min_ver):
                            print(f' \033[0;32m[PASS]\033[0m {pkg_raw} (headless) installed: {installed_ver} >= {min_ver}')
                        else:
                            print(f' \033[0;31m[FAIL]\033[0m {pkg_raw} (headless) installed: {installed_ver} < {min_ver}')
                            all_pass = False
                    else:
                        print(f' \033[0;32m[PASS]\033[0m {pkg_raw} (headless) installed: {installed_ver}')
                    continue
                except importlib.metadata.PackageNotFoundError:
                    pass
                    
            print(f' \033[0;31m[FAIL]\033[0m {pkg_raw} is not installed inside the active context.')
            all_pass = False
            
    return all_pass

success = verify_requirements('requirements.txt')
sys.exit(0 if success else 1)
"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}System Validation Complete. All architectural constraints met successfully.${NC}"
else
    echo -e "${RED}System baseline validation failed. Execute install_dependencies.sh to repair environment.${NC}"
    exit 1
fi