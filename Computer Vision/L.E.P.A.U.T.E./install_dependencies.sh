#!/bin/bash
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}Initializing L.E.P.A.U.T.E. Local Isolation Build Environment...${NC}"

if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Fatal Error: python3 is not installed or not in system PATH.${NC}"
    exit 1
fi

if [ ! -d "venv" ]; then
    echo "Constructing virtual environment (venv)..."
    python3 -m venv venv
fi

echo "Activating virtual environment context..."
source venv/bin/activate

echo "Upgrading base installer abstraction kits (pip, setuptools, wheel)..."
python3 -m pip install --upgrade pip setuptools wheel

if [ ! -f "requirements.txt" ]; then
    echo -e "${RED}Fatal Error: requirements.txt not found in the execution directory.${NC}"
    exit 1
fi

echo "Installing dependency matrix from requirements.txt..."
python3 -m pip install -r requirements.txt

echo -e "${GREEN}L.E.P.A.U.T.E. Dependencies verified and deployment completed successfully.${NC}"