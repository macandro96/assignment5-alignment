import argparse
import os
from itertools import product

import yaml

from cs336_alignment.logger import WandBLogger
from cs336_alignment.sft.trainer import Trainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_path", type=str, required=True, help='cfg path')
    parser.add_argument("--grid_idx", type=int, required=True, help='grid idx')
    args = parser.parse_args()
    
    # setup grid
    lrs = [2e-6, 2e-5, 2e-4]
    sample_size = [128, 256, 512, 1024, 0]
    
    grid_cfgs = list(product(lrs, sample_size))
    
    lr, sample_rows = grid_cfgs[args.grid_idx]
    
    with open(args.cfg_path, "r") as f:
        config = yaml.safe_load(f)
    
    # override config with grid config
    config['optimizer']['lr'] = lr
    config['data']['sample_rows'] = sample_rows
    ckpt_dir = config['training']['ckpt_dir']
    config['training']['ckpt_dir'] = os.path.join(ckpt_dir, f"{args.grid_idx}")
    run_name = config['wandb']['run_name']
    config['wandb']['run_name'] = f"{run_name}-grid={args.grid_idx}"
    
    
    print(config)

    trainer = Trainer(config, logger=WandBLogger(config))

    trainer.train()
    
    
        