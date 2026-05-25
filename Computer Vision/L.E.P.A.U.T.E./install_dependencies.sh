#!/bin/bash
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

echo "Initializing L.E.P.A.U.T.E. Local Isolation Build Environment..."

if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

source venv/bin/activate

echo "Upgrading base installer abstraction kits..."
pip install --upgrade pip setuptools wheel

echo "Installing specified geometric Deep Learning backbones..."
pip install torch>=2.3.0 torchvision>=0.18.0 opencv-python>=4.8.0 numpy>=1.24.0

echo "Installing advanced vision-language and flow modules..."
pip install transformers>=4.40.0 pytorch-metric-learning>=2.0.0 pillow>=10.0.0 timm>=0.9.0

if [ $? -eq 0 ]; then
    echo -e "${GREEN}L.E.P.A.U.T.E. Dependencies verified and deployment completed successfully.${NC}"
else
    echo -e "${RED}Dependency orchestration tree injection failed.${NC}"
    exit 1
fi