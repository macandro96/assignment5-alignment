"""Grid search over epochs_per_rollout and train_batch_size keeping the rest constant."""
import argparse
import os
from itertools import product

import yaml

from cs336_alignment.logger import WandBLogger
from cs336_alignment.rl.rl_trainer import RLTrainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_path", type=str, required=True, help='cfg path')
    parser.add_argument("--grid_idx", type=int, required=True, help='grid idx')
    args = parser.parse_args()
    
    # setup grid
    train_batch_size_grid = [128, 256, 512]
    epochs_per_rollout_grid = [1, 2, 4]
    
    grid_cfgs = list(product(train_batch_size_grid, epochs_per_rollout_grid))
    
    train_batch_size, epochs_per_rollout = grid_cfgs[args.grid_idx]
    
    batch_size_per_epoch = 2  # set grad accum steps based on this
    
    
    with open(args.cfg_path, "r") as f:
        config = yaml.safe_load(f)
    
    # override config with grid config
    ckpt_dir = config['training']['ckpt_dir']
    config['training']['ckpt_dir'] = os.path.join(ckpt_dir, f"{args.grid_idx}")
    run_name = config['wandb']['run_name']
    config['wandb']['run_name'] = f"{run_name}-grid={args.grid_idx}"
    
    # set config vars
    config["training"]["rl"]["train_batch_size"] = train_batch_size
    config["training"]["rl"]["epochs_per_rollout"] = epochs_per_rollout
    config["training"]["gradient_accumulation_steps"] = train_batch_size // batch_size_per_epoch
    
    
    print(config)

    trainer = RLTrainer(config, logger=WandBLogger(config))

    trainer.train()
    