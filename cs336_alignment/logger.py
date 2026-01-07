import wandb


class WandBLogger:
    def __init__(self, cfg: dict):
        # initialize wandb using existing helper
        config_keys_to_dump = [
            "model",
            "optimizer",
            "training",
            "data",
        ]
        config_to_dump = {}
        for key in config_keys_to_dump:
            if key in cfg:
                config_to_dump[key] = cfg[key]

        wandb_cfg = cfg['wandb']
        wandb.init(
            project=wandb_cfg['project'],
            name=wandb_cfg.get('run_name'),
            config=config_to_dump
        )
    
        # Setup wandb metrics
        wandb.define_metric("train_step")  # the x‑axis for training
        wandb.define_metric("eval_step")  # the x‑axis for evaluation
        
        # everything that starts with train/ is tied to train_step
        wandb.define_metric("train/*", step_metric="train_step")
        # everything that starts with eval/ is tied to eval_step
        wandb.define_metric("eval/*", step_metric="eval_step")


    def log(self, metrics: dict, step: int | None = None, step_type: str = "train"):
        """Log metrics to wandb.

        - `step_type` should be either 'train' or 'eval' and controls which step
          key is added ('train_step' or 'eval_step').
        """
        data = dict(metrics)
        if step is not None:
            if step_type == "train":
                data["train_step"] = step
            else:
                data["eval_step"] = step
        wandb.log(data)
