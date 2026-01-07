import argparse
import os
from collections import defaultdict
from typing import Callable

import numpy as np
import pandas as pd
import vllm
from vllm import LLM, SamplingParams

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.utils import generate_data, load_math_dataset


def evaluate_vllm(
    vllm_model: LLM,
    prompts: list[str],
    eval_sampling_params: SamplingParams,
    answers: list[str],
    reward_fn: Callable[[str, str], dict[str, float]],
    dump_path: str | None = None,
    print_metrics: bool = False,
) -> dict:
    """Evaluate a language model on a list of prompts, compute evaluation metrics, and serialize results to disk."""
    outputs = generate_data(
        vllm_model=vllm_model,
        prompts=prompts,
        gen_sampling_params=eval_sampling_params,
    )
    results = defaultdict(list)
    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text
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

    if dump_path:
        results["prompts"] = prompts
        results["answers"] = answers

        data_dump = pd.DataFrame(results)
        data_dump.to_json(dump_path, orient="records", lines=True)
    
    return {
        key: np.mean(value)
        for key, value in results.items()
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-Math-1.5B",
        help="The name of the vLLM model to evaluate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature for evaluation.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p sampling parameter for evaluation.",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=1024,
        help="Maximum generation length for evaluation.",
    )
    parser.add_argument(
        "--dump_path", type=str, default=None, help="Path to dump evaluation results."
    )
    args = parser.parse_args()

    ds = load_math_dataset(split="test", prompt_name="r1_zero.prompt")
    os.makedirs(
        os.path.dirname(args.dump_path), exist_ok=True
    ) if args.dump_path else None

    vllm_model = LLM(model=args.model_name)
    eval_sampling_params = SamplingParams(
        temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens
    )

    prompts = ds["prompt"].tolist()
    answers = ds["solution"].tolist()

    evaluate_vllm(
        vllm_model=vllm_model,
        prompts=prompts,
        eval_sampling_params=eval_sampling_params,
        answers=answers,
        reward_fn=r1_zero_reward_fn,
        dump_path=args.dump_path,
    )
    vllm.distributed.parallel_state.destroy_distributed_environment()
