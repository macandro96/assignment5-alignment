import argparse
import os

import numpy as np
import torch
from tqdm import tqdm
from vllm import SamplingParams

from cs336_alignment import modeling
from cs336_alignment.dataset import ReplayBuffer
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.eval import VLLM_Eval
from cs336_alignment.utils import (
    compute_group_normalized_rewards,
    generate_data,
    get_grad_norm,
    get_response_log_probs,
    grpo_microbatch_train_step,
    load_math_dataset,
    tokenize_prompt_and_output,
)


class RLTrainer:
    def __init__(
        self,
        cfg: dict,
        policy_device: str | None = None,
        vllm_device: str | None = None,
        logger=None
    ):
        self.cfg = cfg
        
        self.policy_device = policy_device or (
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        self.vllm_device = vllm_device or (
            "cuda:1" if torch.cuda.device_count() > 1 else self.policy_device
        )
        
        self.logger = logger
        
    def setup(self):
        # setup rl policies
        self.policy, self.tokenizer = modeling.load_model(
            self.cfg["model"],
            device=self.policy_device
        )
        
        if self.cfg.get("vllm_eval"):
            self.vllm_model = VLLM_Eval(
                gpu_memory_utilization=self.cfg["vllm_eval"]["gpu_memory_utilization"],
                seed=self.cfg["vllm_eval"]["seed"],
                model_id=self.cfg["model"]["name"],
                device=self.vllm_device,
                monkey_patch_init=True,
                dtype=modeling._get_dtype(self.cfg["model"]["dtype"]),
            )
            
            self.sampling_params = SamplingParams(
                temperature=self.cfg['vllm_eval']["sampling_temperature"],
                min_tokens=self.cfg["vllm_eval"]["sampling_min_tokens"],
                max_tokens=self.cfg["vllm_eval"]["sampling_max_tokens"],
                logprobs=1
            )
        
        self.replay_buffer = ReplayBuffer()
        
        # it is on policy if train_batch_size == rollout_batch_size
        self.on_policy = self.cfg['training']['rl']['train_batch_size'] == \
            self.cfg['training']['rl']['rollout_batch_size']
        
        opt_scheduler = modeling.load_optimizer_and_scheduler(
            optim_cfg=self.cfg["optimizer"],
            params=self.policy.parameters(),
            total_training_steps=None,
        )
        self.optimizer = opt_scheduler["optimizer"]
    
    
    def obtain_logits_response_mask(
        self,
        prompts: list[str],
        rollouts: list[str]
    ):
        """Obtain logits for generated rollout per prompt."""
        tokenized = tokenize_prompt_and_output(
            prompt_strs=prompts,
            output_strs=rollouts,
            tokenizer=self.tokenizer,
        )
        logprobs_entropy = get_response_log_probs(
            model=self.policy,
            input_ids=tokenized["input_ids"].to(self.policy_device),
            labels=tokenized["labels"].to(self.policy_device),
            attention_mask=tokenized["attention_mask"].to(self.policy_device),
            return_token_entropy=True
        )
        
        return {
            "log_probs": logprobs_entropy["log_probs"],
            "token_entropy": logprobs_entropy["token_entropy"],
            "response_mask": tokenized["response_mask"].to(self.policy_device)
        }
    
    def rollouts(
        self,
        sample_size: int,
        group_size: int,
        split: str = "train",
    ):
        # sample batch of questions
        sample_dataset = load_math_dataset(
            split=split,
            sample=sample_size,
        )  # dataframe with prompt and solution
        
        # generate rollouts
        prompts = sample_dataset['prompt'].tolist()
        solutions = sample_dataset['solution'].tolist()

        prompts_group = sum([[prompt] * group_size for prompt in prompts], [])
        sols_group = sum([[sol] * group_size for sol in solutions], [])
        
        rollouts, logprobs = generate_data(
            self.vllm_model.llm,
            prompts_group,
            self.sampling_params,
            flatten_multiple=True,
            return_log_probs=True
        )
        # token length of rollouts
        rollout_len= [
            len(lp)
            for lp in logprobs
        ]

        # obtain advantages and rewards
        advantages, rewards, metadata = compute_group_normalized_rewards(
            reward_fn=r1_zero_reward_fn,
            rollout_responses=rollouts,
            repeated_ground_truths=sols_group,
            group_size=group_size,
            advantage_eps=float(self.cfg['training']['rl']['advantage_eps']),
            normalize_by_std=self.cfg['training']['rl']['use_std_normalization']
        )
        
        if isinstance(advantages, torch.Tensor):

            advantages = advantages.tolist()
        if isinstance(rewards, torch.Tensor):
            rewards = rewards.tolist()
        
        return {
            "prompts": prompts_group,  # list[str]
            "solutions": sols_group,   # list[str]
            "rollouts": rollouts,      # list[str]
            "logprobs": logprobs,       # list[torch.Tensor]
            "advantages": advantages,  # list[float]
            "rewards": rewards,  # list[float]
            "rollout_len": rollout_len,
        }
    
    def train_step(self):
        train_batch_size = self.cfg['training']['rl']['train_batch_size']

        # update old policy
        self.vllm_model.load_policy_into_vllm_instance(
            policy=self.policy
        )
        log_data = {
            "rewards": [],
            "advantages": [],
            "rollout_len": [],
        }
        group_size = self.cfg["training"]["rl"]["group_size"]
        sample_size = self.cfg["training"]["rl"]["rollout_batch_size"] // \
                group_size
            
        while len(self.replay_buffer) < train_batch_size:
            # generate group num of rollouts rollouts per rollout_batch_size prompt
            # contains rollout_batch_size x group_size prompts and rollouts
            rollout_data = self.rollouts(sample_size=sample_size, group_size=group_size)

            advantages = rollout_data["advantages"]  # list[float]
            rewards = rollout_data["rewards"]  # list[float]
            rollout_len = rollout_data["rollout_len"]  # list[int]
            
            # do another rollout for test
            with torch.no_grad():
                micro_batch_size = 1
                log_probs = []
                for idx in range(0, len(rollout_data["prompts"]), micro_batch_size):
                    log_prob_idx = self.obtain_logits_response_mask(
                        prompts=rollout_data["prompts"][idx: idx + micro_batch_size],
                        rollouts=rollout_data["rollouts"][idx: idx + micro_batch_size],
                    )["log_probs"]
                    
                    log_probs.append(log_prob_idx)

            self.replay_buffer.push(
                [
                    {
                        "prompt": prompt,
                        "advantage": advantage,
                        "reward": reward,
                        "rollout": rollout,
                        "logprob": logprob  # list of torch tensors
                    }
                    for prompt, advantage, reward, rollout, logprob in zip(
                        rollout_data['prompts'],
                        advantages,
                        rewards,
                        rollout_data['rollouts'],
                        # rollout_data["logprobs"],
                        log_probs,
                    )
                ]
            )
            log_data["rewards"].extend(rewards)
            log_data["advantages"].extend(advantages)
            log_data["rollout_len"].extend(rollout_len)

        if self.logger:
            log_dict = {
                    "train/generated_rewards_mean": np.mean(log_data["rewards"]),
                    "train/generated_advantages_mean": np.mean(log_data["advantages"]),
                    "train/generated_rollouts_len": np.mean(log_data["rollout_len"])
                }
            self.logger.log(
                log_dict
            )
            print(f"Log dict = {log_dict}")
        
        train_batch = self.replay_buffer.pop(train_batch_size)
        num_epochs = self.cfg['training']["rl"]['epochs_per_rollout_batch']
        
        # cannot feed in entire batch; need to split into micro_batch using grad_accum_steps
        grad_accum_steps = self.cfg['training']['gradient_accumulation_steps']
        micro_batch_size = len(train_batch) // grad_accum_steps
        
        for epoch_idx in range(num_epochs):
            train_log_data = {
                "train/epoch_loss": 0.0,
                "train/token_entropy": 0.0,
                "train/grad_norm": 0,
                "train/train_rewards": 0,
            }
            for idx in range(0, len(train_batch), micro_batch_size):
                
                # select microbatch
                micro_batch = train_batch[idx: idx + micro_batch_size]

                # obtain logits and response_mask using policy model for microbatch
                logits_response_mask = self.obtain_logits_response_mask(
                    prompts=[batch["prompt"] for batch in micro_batch],
                    rollouts=[batch["rollout"] for batch in micro_batch]
                )  # bsz x seq
                policy_log_probs, response_mask = logits_response_mask["log_probs"], \
                    logits_response_mask["response_mask"]
                
                token_entropy = logits_response_mask["token_entropy"]
                token_entropy = token_entropy.float().mean()
                
                # policy_log_probs is of shape bsz x (prompt + gen tokens)
                # old_log_probs is of shape (bsz x (gen tokens))
                # pad old_log_probs with 0 for prompt + pad tokens

                batch_old_log_probs = [batch["logprob"] for batch in micro_batch]
                batch_lens = [batch["logprob"].shape[1] for batch in micro_batch]
                max_batch_len = max(batch_lens)
                # batch with padding
                for i in range(len(batch_old_log_probs)):
                    balance_lens = max_batch_len - batch_lens[i]
                    zeros = torch.zeros(size=(1, balance_lens), device=batch_old_log_probs[i].device)

                    batch_old_log_probs[i] = torch.cat([batch_old_log_probs[i], zeros], dim=-1)

                old_log_probs = torch.concat(batch_old_log_probs, dim=0)

                # for i in range(len(old_log_probs)):
                #     nonzero_idx = torch.nonzero(response_mask[i]).squeeze()
                #     assert len(batch_old_log_probs[i]) == len(nonzero_idx), \
                #         f"Length mismatch: {len(batch_old_log_probs[i])} vs {len(nonzero_idx)}"
                #     src_tensor = batch_old_log_probs[i].to(old_log_probs.device).to(old_log_probs.dtype)
                #     # Calculate the minimum length to avoid crash
                #     min_len = min(len(nonzero_idx), len(src_tensor))
                #     old_log_probs[i, nonzero_idx[:min_len]] = src_tensor[:min_len]
                # import pdb; pdb.set_trace()
                # extract advantages and rewards for micro_batch
                advantages = torch.tensor(
                    [
                        batch["advantage"] for batch in micro_batch
                    ],
                    device=self.policy_device
                ).unsqueeze(-1)  # bsz x 1
                
                raw_rewards = torch.tensor(
                    [
                        batch["reward"] for batch in micro_batch
                    ],
                    device=self.policy_device
                ).unsqueeze(-1)  # bsz x 1
                
                # do train step
                loss, metadata = grpo_microbatch_train_step(
                    policy_log_probs=policy_log_probs,
                    response_mask=response_mask,
                    gradient_accumulation_steps=grad_accum_steps,
                    loss_type=self.cfg["training"]["rl"]["loss_type"],
                    raw_rewards=raw_rewards,
                    advantages=advantages,
                    old_log_probs=old_log_probs,
                    cliprange=self.cfg["training"]["rl"]["cliprange"],
                    length_normalize=self.cfg["training"]["rl"]["length_normalize"]
                )
                    
                # accumulate log data
                train_log_data["train/epoch_loss"] += loss.item()
                train_log_data["train/token_entropy"] += (token_entropy.item() / grad_accum_steps)
                train_log_data["train/train_rewards"] += (torch.sum(raw_rewards))

            train_log_data["train/grad_norm"] = get_grad_norm(self.policy)
            train_log_data["train/train_rewards"] /= len(train_batch)

            torch.nn.utils.clip_grad_value_(
                self.policy.parameters(), clip_value=1.0
            )
            
            # optimizer update
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            
            if self.logger:
                self.logger.log(
                    train_log_data,
                    step=self.grpo_step * num_epochs + epoch_idx,
                    step_type="train"
                )
            print(f"Train log data = {train_log_data}")

    def eval_step(self):
        rollout_data = self.rollouts(
            sample_size=self.cfg["training"]["eval_sample"],
            group_size=1,
            split="test"
        )
        
        # mean rewards
        mean_reward = np.mean(rollout_data["rewards"])
        mean_rollout_len = np.mean(rollout_data["rollout_len"])
        log_data = {
            "eval/test_mean_reward": mean_reward,
            "eval/test_mean_rollout_len": mean_rollout_len,
        }
        if self.logger:
            self.logger.log(
                log_data,
                step_type="eval",
                step=self.grpo_step,
            )
        print(f"Eval log: {log_data}")

    def train(self):
        self.setup()
        
        n_steps = self.cfg['training']['rl']['n_grpo_steps']
        
        for step in tqdm(range(n_steps)):
            self.grpo_step = step  # global access for logging
            self.train_step()

            if self.grpo_step % self.cfg["training"]["eval_log_steps"] == 0:
                self.eval_step()

            if self.grpo_step % self.cfg["training"]["save_steps"] == 0:
                out_dir = os.path.join(self.cfg["training"]["ckpt_dir"], f"grpo_step={self.grpo_step}")
                modeling.save_model_tokenizer(self.policy, self.tokenizer, out_dir)
            
            self.grpo_step += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_path", type=str, help="path to config")
    args = parser.parse_args()
    
    import yaml

    with open(args.cfg_path, "r") as f:
        config = yaml.safe_load(f)
    
    from cs336_alignment.logger import WandBLogger
    trainer = RLTrainer(config, logger=WandBLogger(config))
    trainer.train()
