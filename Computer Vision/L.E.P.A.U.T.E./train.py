import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import random_split, Dataset

try:
    from module import LepauteConfig, SE3ResidualRefiner, EquivariantDataset, train_sequence_loop
except ImportError as e:
    print(f"CRITICAL: Failed to import LEPAUTE infrastructure components from 'module.py': {e}")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] LEPAUTE_TRAIN: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("LEPAUTE.OfflineTraining")


def set_reproducibility_seeds(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_data_splits(dataset: Dataset, split_ratio: float = 0.2) -> Tuple[Dataset, Optional[Dataset]]:
    dataset_size = len(dataset)
    if dataset_size < 8:
        logger.warning(
            f"Dataset size ({dataset_size}) is insufficient for validation partitioning. "
            f"Deploying entire allocation to the training track."
        )
        return dataset, None
    
    val_len = int(dataset_size * split_ratio)
    train_len = dataset_size - val_len
    
    generator = torch.Generator().manual_seed(42)
    return random_split(dataset, [train_len, val_len], generator=generator)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Production Orchestration for training LEPAUTE SE(3) Subsystems.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--dataset_dir", type=str, required=True, help="Root path containing lepaute_data.json and frames/")
    parser.add_argument("--epochs", type=int, default=15, help="Maximum number of processing epochs.")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Target path for state saving.")
    parser.add_argument("--device", type=str, default=None, help="Force override configuration compute target (e.g., 'cuda', 'mps', 'cpu').")
    parser.add_argument("--no_compile", action="store_true", help="Disable PyTorch 2.x torch.compile() optimization step.")
    parser.add_argument("--seed", type=int, default=42, help="Stochastic initialization seed value.")
    
    args = parser.parse_args()

    set_reproducibility_seeds(args.seed)

    data_root = Path(args.dataset_dir)
    json_path = data_root / "lepaute_data.json"
    frames_path = data_root / "frames"
    ckpt_root = Path(args.checkpoint_dir)
    
    if not json_path.exists():
        logger.error(f"Execution Aborted: Expected dataset manifest file not found at: {json_path}")
        sys.exit(1)
        
    if not frames_path.is_dir():
        logger.error(f"Execution Aborted: Expected image sequence target context directory not found at: {frames_path}")
        sys.exit(1)

    try:
        ckpt_root.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error(f"System IO Error: Unable to create checkpoint destination directory {ckpt_root}: {e}")
        sys.exit(1)
    
    config = LepauteConfig()
    
    if args.device is not None:
        config.device = args.device
    if args.no_compile:
        config.use_compiler = False

    logger.info(f"LEPAUTE Engine Configuration Initialized. Targets: Device={config.device} | Compiler={config.use_compiler}")

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            stored_records = json.load(f)
    except json.JSONDecodeError as jde:
        logger.error(f"Corruption Detected: Failed parsing structured json sequence file from manifest: {jde}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"File Access Error: Encountered failure attempting to open target data payload manifest: {e}")
        sys.exit(1)
    
    logger.info(f"Dataset successfully compiled. Found {len(stored_records)} active monocular sequence blocks.")
    
    try:
        dataset = EquivariantDataset(data_list=stored_records, config=config, data_dir=str(data_root))
    except Exception as e:
        logger.error(f"Initialization Exception: Data structure initialization failure within EquivariantDataset: {e}")
        sys.exit(1)
        
    train_set, val_set = get_data_splits(dataset)
    
    logger.info("Instantiating Deep SE(3) Residual Refiner Subsystem architecture.")
    try:
        model = SE3ResidualRefiner(config=config)
    except Exception as e:
        logger.error(f"Architecture Generation Error: Failed to construct neural network graph layout: {e}")
        sys.exit(1)
    
    logger.info(f"Optimization track initialized. Syncing checkpoints output path to: {ckpt_root.resolve()}")
    
    try:
        train_loss, val_loss = train_sequence_loop(
            model=model,
            train_dataset=train_set,
            val_dataset=val_set,
            config=config,
            epochs=args.epochs,
            checkpoint_dir=str(ckpt_root)
        )
        logger.info(f"Optimization Sequence Concluded Successfully. Training Loss: {train_loss:.6f} | Validation Loss: {val_loss:.6f}")
        
    except KeyboardInterrupt:
        logger.warning("Optimization sequence execution forcibly interrupted by operational operator signal (SIGINT). Clean exit enforced.")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Runtime Operational Critical Failure: Execution context interrupted during deep tracking loop: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()