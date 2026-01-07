from collections import defaultdict
from typing import Callable
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM
from vllm import LLM, SamplingParams
from vllm.model_executor import set_random_seed as vllm_set_random_seed

from cs336_alignment.utils import generate_data


class VLLM_Eval:
    def __init__(
        self,
        model_id: str,
        device: str,
        seed: int,
        gpu_memory_utilization: float = 0.85,
        dtype: torch.dtype = torch.bfloat16,
        monkey_patch_init: bool = False,
    ):
        vllm_set_random_seed(seed)
        if monkey_patch_init:
            self.llm = VLLM_Eval.init_vllm(
                model_id=model_id,
                device=device,
                gpu_memory_utilization=gpu_memory_utilization,
                dtype=dtype,
            )

        else:
            self.llm = LLM(
                model_id,
                device=device,
                gpu_memory_utilization=gpu_memory_utilization,
                dtype=dtype,
                enable_prefix_caching=True
            )
        
    @staticmethod
    def init_vllm(model_id: str, device: str, gpu_memory_utilization: float = 0.85, dtype: torch.dtype = torch.bfloat16):
        """Start the inference process, here we use vLLM to hold a model on a GPU separate from the policy."""

        # Monkeypatch from TRL:
        # https://github.com/huggingface/trl/blob/
        # 22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py
        # Patch vLLM to make sure we can
        # (1) place the vLLM model on the desired device (world_size_patch) and
        # (2) avoid a test that is not designed for our setting (profiling_patch).
        world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
        profiling_patch = patch("vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling", return_value=None)

        with world_size_patch, profiling_patch:
            return LLM(
                model=model_id,
                device=device,
                dtype=dtype,
                enable_prefix_caching=True,
                gpu_memory_utilization=gpu_memory_utilization,
            )

    def load_policy_into_vllm_instance(self, policy: AutoModelForCausalLM):
        """Copied from https://github.com/huggingface/trl/blob/22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py#L670."""
        state_dict = policy.state_dict()
        llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
        llm_model.load_weights(state_dict.items())

    def evaluate_vllm(
        self,
        prompts: list[str],
        eval_sampling_params: SamplingParams,
        answers: list[str],
        reward_fn: Callable[[str, str], dict[str, float]],
        dump_path: str | None = None,
        print_metrics: bool = False,
        num_samples=1,
    ) -> dict:
        """Evaluate a language model on a list of prompts, compute evaluation metrics, and serialize results to disk.
        Return evaluation metrics and generated text.
        """

        prompts = sum([[prompt] * num_samples for prompt in prompts], [])
        answers = sum([[answer] * num_samples for answer in answers], [])

        outputs = generate_data(
            vllm_model=self.llm,
            prompts=prompts,
            gen_sampling_params=eval_sampling_params,
            flatten_multiple=True
        )

        results = defaultdict(list)

        for i, generated_text in enumerate(outputs):
            reward = reward_fn(generated_text, answers[i])

            # format reward, answer reward, reward
            results["generated_text"].append(generated_text)
            for key, value in reward.items():
                results[key].append(value)

        if print_metrics:
            print("\n===== Evaluation Results ======\n")
            for key in results:
                if key.endswith("reward"):
                    print(f"{key}: {np.mean(results[key]):.4f}")
            print("\n==============================\n")

        results["prompts"] = prompts
        results["answers"] = answers

        if dump_path:    
            data_dump = pd.DataFrame(results)
            data_dump.to_json(dump_path, orient="records", lines=True)
        
        return results
