"""Compare diff. lrs with on-policy"""
import argparse
import os

import yaml

from cs336_alignment.logger import WandBLogger
from cs336_alignment.rl.rl_trainer import RLTrainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_path", type=str, required=True, help='cfg path')
    parser.add_argument("--grid_idx", type=int, required=True, help='grid idx')
    args = parser.parse_args()
    
    # setup grid
    lrs = [1e-8, 1e-7, 1e-6, 1e-5, 5e-5, 1e-4]
    
    lr = lrs[args.grid_idx]
    
    with open(args.cfg_path, "r") as f:
        config = yaml.safe_load(f)
    
    # override config with grid config
    config['optimizer']['lr'] = lr
    ckpt_dir = config['training']['ckpt_dir']
    config['training']['ckpt_dir'] = os.path.join(ckpt_dir, f"{args.grid_idx}")
    run_name = config['wandb']['run_name']
    config['wandb']['run_name'] = f"{run_name}-grid={args.grid_idx}"
    
    
    print(config)

    trainer = RLTrainer(config, logger=WandBLogger(config))

    trainer.train()
    