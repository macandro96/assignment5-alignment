import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from cs336_alignment import modeling
from cs336_alignment.dataset import sft_dataloader
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.eval import VLLM_Eval
from cs336_alignment.utils import (
    get_response_log_probs,
    load_math_dataset,
    sft_microbatch_train_step,
)


class Trainer:
    def __init__(
        self,
        cfg: dict,
        policy_device: str | None = None,
        vllm_device: str | None = None,
        logger=None,
    ):
        self.cfg = cfg
        self.policy_device = policy_device or (
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        self.vllm_device = vllm_device or (
            "cuda:1" if torch.cuda.device_count() > 1 else self.policy_device
        )
        self.logger = logger

        # placeholders
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        self.scheduler = None
        self.vllm_model = None

    def setup(self):
        # model & tokenizer
        self.model, self.tokenizer = modeling.load_model(
            self.cfg["model"], device=self.policy_device
        )

        # dataloader
        self.dataloader = sft_dataloader(
            data_path=self.cfg["data"]["data_path"],
            batch_size=self.cfg["data"]["batch_size"],
            num_workers=self.cfg["data"]["num_workers"],
            tokenizer=self.tokenizer,
            sample_rows=self.cfg["data"].get("sample_rows", 0),
        )

        # eval data
        self.sample_dataset = load_math_dataset(
            sample=self.cfg["data"].get("eval_sample", 0)
        )

        # training steps & optimizer
        epochs = self.cfg["training"]["epochs"]
        grad_accum_steps = self.cfg["training"]["grad_accum_steps"]
        steps_per_epoch = len(self.dataloader) // grad_accum_steps
        total_training_steps = steps_per_epoch * epochs

        opt_scheduler = modeling.load_optimizer_and_scheduler(
            optim_cfg=self.cfg["optimizer"],
            params=self.model.parameters(),
            total_training_steps=total_training_steps,
        )
        self.optimizer = opt_scheduler["optimizer"]
        self.scheduler = opt_scheduler["scheduler"]

        # vllm eval instance
        if self.cfg.get("vllm_eval"):
            self.vllm_model = VLLM_Eval(
                **self.cfg["vllm_eval"],
                model_id=self.cfg["model"]["name"],
                device=self.vllm_device,
                monkey_patch_init=True,
                dtype=modeling._get_dtype(self.cfg["model"]["dtype"]),
            )
            self.all_test = load_math_dataset(sample=0)

            from vllm import SamplingParams
            self.sampling_params = SamplingParams(
                temperature=1.0, top_p=0.9, max_tokens=1024
            )

    def evaluate_sample(self, model=None):
        model = model or self.model
        prompts, sols = (
            self.sample_dataset["prompt"].tolist(),
            self.sample_dataset["solution"].tolist(),
        )

        # tokenizer padding side
        tok_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        tokenized = self.tokenizer(prompts, return_tensors="pt", padding="longest").to(
            self.policy_device
        )
        self.tokenizer.padding_side = tok_padding_side

        model.eval()
        with torch.no_grad():
            outputs = model.generate(
                **tokenized,
                max_new_tokens=1024,
                temperature=None,
                pad_token_id=self.tokenizer.pad_token_id,
            ).to("cpu")
        model.train()

        input_len = tokenized['input_ids'].shape[1]
        generated_tokens = outputs[:, input_len:]
        generated_text = [
            self.tokenizer.decode(t, skip_special_tokens=True)
            for t in generated_tokens
        ]

        rewards = [
            r1_zero_reward_fn(gen_text, sol)
            for gen_text, sol in zip(generated_text, sols)
        ]
        format_rewards = [r['format_reward'] for r in rewards]
        answer_rewards = [r['answer_reward'] for r in rewards]

        return {
            "prompts": prompts,
            "solution": sols,
            "generated_text": generated_text,
            "format_reward": format_rewards,
            "answer_reward": answer_rewards,
        }

    def eval_step(
        self,
        global_step
    ):
        # eval on sample test rows
        eval_out = self.evaluate_sample(self.model)
        eval_table = pd.DataFrame(eval_out)
        eval_table['global_step'] = global_step
        
        # vllm evaluation
        if self.vllm_model:
            self.vllm_model.load_policy_into_vllm_instance(self.model)
            metrics = self.vllm_model.evaluate_vllm(
                prompts=self.all_test["prompt"].tolist(),
                eval_sampling_params=self.sampling_params,
                answers=self.all_test["solution"].tolist(),
                reward_fn=r1_zero_reward_fn,
                print_metrics=True
            )

        else:
            metrics = {"format_reward": [0.0], "answer_reward": [0.0]}

        if self.logger:
            self.logger.log(
                {
                    "eval/generations": eval_table,
                    "eval/sample_avg_format_reward": eval_table["format_reward"].mean(),
                    "eval/sample_avg_answer_reward": eval_table["answer_reward"].mean(),
                    "eval/avg_format_reward": np.mean(metrics['format_reward']),
                    "eval/avg_answer_reward": np.mean(metrics['answer_reward']),
                },
                step=global_step,
                step_type="eval",
            )

        
    def train(self):
        # setup
        self.setup()
        cfg = self.cfg

        # prepare for training
        epochs = cfg["training"]["epochs"]
        grad_accum_steps = cfg["training"]["grad_accum_steps"]
            
        for epoch in tqdm(range(epochs)):
            for step, batch in tqdm(
                enumerate(self.dataloader), total=len(self.dataloader)
            ):
                global_step = epoch * len(self.dataloader) + step

                response_log_probs = get_response_log_probs(
                    model=self.model,
                    input_ids=batch["input_ids"].to(self.policy_device),
                    labels=batch["labels"].to(self.policy_device),
                )

                loss, _ = sft_microbatch_train_step(
                    policy_log_probs=response_log_probs["log_probs"],
                    response_mask=batch["response_mask"].to(self.policy_device),
                    gradient_accumulation_steps=grad_accum_steps,
                    normalize_constant=batch['response_mask'].sum().item(),
                )
                current_lr = self.optimizer.param_groups[0]["lr"]

                if (step + 1) % grad_accum_steps == 0:
                    torch.nn.utils.clip_grad_value_(
                        self.model.parameters(), clip_value=1.0
                    )
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                if (global_step) % cfg["training"]["logging_steps"] == 0:
                    print(f"Loss: {loss.item()}")
                    if self.logger:
                        self.logger.log(
                            {"train/loss": loss.item(), "train/lr": current_lr},
                            step=global_step,
                            step_type="train",
                        )
                
                if (global_step) % cfg['training']['eval_log_steps'] == 0:
                    self.eval_step(global_step=global_step)

                if (global_step) % cfg["training"]["save_steps"] == 0:
                    ckpt_dir = cfg["training"]["ckpt_dir"]
                    out_dir = os.path.join(ckpt_dir, f"step={global_step}")
                    modeling.save_model_tokenizer(self.model, self.tokenizer, out_dir)
                

        # final eval
        self.eval_step(global_step=global_step)
        # final save
        ckpt_dir = cfg["training"]["ckpt_dir"]
        out_dir = os.path.join(ckpt_dir, "final_model")
        modeling.save_model_tokenizer(self.model, self.tokenizer, out_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg_path", type=str, required=True, help="path to config file"
    )
    args = parser.parse_args()

    import yaml

    with open(args.cfg_path, "r") as f:
        config = yaml.safe_load(f)

    from cs336_alignment.logger import WandBLogger

    trainer = Trainer(config, logger=WandBLogger(config))

    trainer.train()
