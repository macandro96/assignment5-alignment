import os
from collections import defaultdict
from typing import Callable, Literal

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import PreTrainedModel
from vllm import LLM, SamplingParams

PROMPT_DIR = os.environ.get("PROMPT_DIR", "cs336_alignment/prompts/")
DATA_DIR = os.environ.get("DATA_DIR", "data/math/")


def load_math_dataset(split: str = "test", prompt_name: str = "r1_zero.prompt", sample: int = 0):
    """The MATH dataset has been cached into `MATH` folder. Comprises of train.jsonl and test.jsonl files.

    This function loads the dataset and applies the prompt template to each example.
    """
    dataset = pd.read_json(os.path.join(DATA_DIR, f"{split}.jsonl"), lines=True)
    if sample > 0:
        dataset = dataset.sample(sample)

    prompt_text = open(os.path.join(PROMPT_DIR, prompt_name)).read()
    dataset["prompt"] = dataset["problem"].apply(
        lambda x: prompt_text.format(question=x)
    )

    return dataset

def get_grad_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5

from vllm import LLM, SamplingParams


def generate_data(
    vllm_model: LLM,
    prompts: list[str],
    gen_sampling_params: SamplingParams,
    flatten_multiple: bool = False,
    return_log_probs: bool = False,
) -> tuple[list[str], list[torch.Tensor]] | tuple[list[list[str]], list[torch.Tensor]] | list[str] | list[list[str]]:
    """Generate outputs and logprobs.
    
    Returns:
        A tuple (texts, logprobs)
        - texts: Same structure as before (list of strings or list of lists)
        - logprobs: Parallel list containing the logprob dictionaries for every token.
    """
    
    # 1. CRITICAL: Ensure vLLM actually computes logprobs
    # If not set, we default to 1 (returns logprob of the generated token only)
    if gen_sampling_params.logprobs is None:
        gen_sampling_params.logprobs = 1

    gen_sampling_params.stop = ["</answer>"]
    gen_sampling_params.include_stop_str_in_output = True

    outputs = vllm_model.generate(prompts, sampling_params=gen_sampling_params)

    # Initialize containers
    all_texts = []
    all_logprobs = []

    # Helper to process a single RequestOutput
    def process_output(output):
        """Extracts text and converts logprobs to tensors for a single request."""
        texts = []
        logprobs_tensors = []
        
        for out in output.outputs:
            # 1. Get Text
            texts.append(out.text)
            
            # 2. Get Logprobs
            # out.token_ids: The list of integers representing the generated tokens
            # out.logprobs: A list of dicts. Each dict maps {token_id: LogprobObject}
            
            token_ids = out.token_ids
            logprobs_list = out.logprobs
            
            seq_log_probs = []
            
            for i, token_id in enumerate(token_ids):
                # Retrieve the logprob object for the specific token that was chosen.
                # vLLM returns a dict of top-k logprobs. We look up the one we actually picked.
                if token_id in logprobs_list[i]:
                    val = logprobs_list[i][token_id].logprob
                    seq_log_probs.append(val)
                else:
                    # Fallback: This rarely happens unless logprobs < 1 or vocabulary mismatch
                    # defaulting to -inf or 0.0 might be safer than crashing
                    seq_log_probs.append(0.0) 

            # Create tensor on CPU (to save VRAM until needed)
            logprobs_tensors.append(torch.tensor(seq_log_probs, dtype=torch.float32))
            
        return texts, logprobs_tensors

    # Standard case (n=1)
    for output in outputs:
        texts, logprobs = process_output(output)
        
        if flatten_multiple:
            all_texts.extend(texts)
            all_logprobs.extend(logprobs)
        else:
            # Wrap in list to match original list[list] structure if n=1 but flatten=False
            all_texts.append(texts)
            all_logprobs.append(logprobs)

    if return_log_probs:
        # return text generated along with list of torch tensor logprobs
        return all_texts, all_logprobs

    return all_texts

def tokenize_prompt_and_output(
    prompt_strs, output_strs, tokenizer
) -> dict[str, torch.Tensor]:
    """Tokenize `prompt` and `output` strings. Concatenate them and return tokens, labels and response_mask."""
    prompt_tokens = tokenizer(prompt_strs)["input_ids"]
    output_tokens = tokenizer(output_strs)["input_ids"]

    # compute max of all `prompt_and_output_lens`
    max_len = max(
        [
            len(prompt_tok) + len(output_tok)
            for prompt_tok, output_tok in zip(prompt_tokens, output_tokens)
        ]
    )
    tokenized_outputs = defaultdict(list)
    for prompt_tok, output_tok in zip(prompt_tokens, output_tokens):
        # input_ids
        prompt_output_tok = prompt_tok + output_tok
        balance = max_len - len(prompt_output_tok)
        balance_pad_tokens = [tokenizer.pad_token_id] * balance
        input_ids = prompt_output_tok + balance_pad_tokens
        input_ids = input_ids[:-1]

        # labels
        labels = prompt_output_tok + balance_pad_tokens
        labels = labels[1:]

        # response mask
        response_mask = (
            [0] * (len(prompt_tok) - 1) + [1] * len(output_tok) + [0] * balance
        )
        
        attention_mask = ([1] * len(prompt_output_tok) + [0] * balance)[:-1]

        tokenized_outputs["input_ids"].append(input_ids)
        tokenized_outputs["response_mask"].append(response_mask)
        tokenized_outputs["labels"].append(labels)
        tokenized_outputs["attention_mask"].append(attention_mask)

    for key, value in tokenized_outputs.items():
        tokenized_outputs[key] = torch.tensor(value, dtype=torch.int64)

    return tokenized_outputs


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Entropy over vocab dimensions.
    
    H(p) = -sum(p(x) x log(p(x)))
    p(x_i) = exp(logits(x_i)) / sum(exp(logits(x_j)))
    log(p(x_i)) = logits(x_i) - log(sum(exp(logits(x_j))))
    p(x_i) * log(p(x_i)) = \
        exp(logits(x_i)) / sum(exp(logits(x_j))) * logits(x_i) - log(sum(exp(logits(x_j))))

    Args:
        - logits: un-normalized torch.Tensor of shape batch_size x seq_length x vocab_size
    Returns:
        entropy: torch.Tensor of shape batch_size x seq_length
    """
    probs = F.softmax(logits, dim=-1)  # bsz x seq x vocab
    logprobs = logits - torch.logsumexp(
        logits, dim=-1, keepdim=True
    )  # bsz x seq x vocab
    return -torch.sum(probs * logprobs, dim=-1)


def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
    attention_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Function to obtain per-token conditional log-probabilities from a causal language model.

    Args:
        - model (PreTrainedModel): model to compute logits
        - input_ids (torch.Tensor): tokens of input + response returned by our tokenizer function; shape: batch_size x seq_len
        - labels (torch.Tensor): labels of input + response returned by our tokenizer function; shape: batch_size x seq_len
        - return_token_entropy (bool): return token entropy if set (default: False)

    Returns:
        - dict[str, torch.Tensor]: contains keys
            log_probs: maps to tensor of shape batch_size x seq_length
            token_entropy: maps to tensor of shape batch_size x seq_length
    """
    logits = model(input_ids).logits  # batch_size x seq_len x |V|

    logprobs = F.log_softmax(logits, dim=-1)  # bsz x seq_len x |V|
    logprobs = torch.gather(logprobs, dim=-1, index=labels.unsqueeze(-1)).squeeze(
        -1
    )  # bsz x seq_len

    response_log_probs = {"log_probs": logprobs}
    if return_token_entropy:
        token_entropy = compute_entropy(logits.detach())
        response_log_probs["token_entropy"] = token_entropy

    return response_log_probs


def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalize_constant: float | torch.Tensor,
    dim: int | None = None,
):
    """Will be used to sum out the values in the tensor where mask values are 1. Essentially used to compute
    log probs over non-prompt tokens

    Args:
        - tensor (torch.Tensor): tensor to normalize
        - mask (torch.Tensor): mask to use (same shape as tensor)
        - normalize_constant (float): dividing by this value after summing
        - dim (int | None): dim to sum before normalization. If None, sum over all dims
    """
    sum_vals = torch.sum(tensor * mask, dim=dim)
    return sum_vals / normalize_constant


def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute cross-entropy loss and scale it with `gradient_accumulation_steps`

    Args:
        - policy_log_probs (torch.Tensor): per token log probs from SFT policy model (bsz x seq_length)
        - response_mask (torch.Tensor): 1 for response tokens, 0 otherwise
        - gradient_accumulation_steps (int): grad accumulation steps
        - normalize_constant (float): normalize constant while computing loss
    """
    # compute loss
    loss = -masked_normalize(
        tensor=policy_log_probs,
        mask=response_mask,
        normalize_constant=normalize_constant * gradient_accumulation_steps,
    )  # / (1 * gradient_accumulation_steps)
    loss.backward()
    return loss, {
    }

def compute_group_normalized_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float = 1e-6,
    normalize_by_std: bool = True
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Calculates reward for every rollout response and normalizes it within their groups.
    
    Args:
        - reward_fn: reward function
        - rollout_responses: list of responses of size group_size * len(prompts)
        - repeated_ground_truths: list of ground truths of size group_size * len(prompts)
        - group_size: size of group
        - advantage_eps: constant to avoid division by zero in normalization
        - normalize_by_std: True if normalize by std else False
    
    Returns:
        advantages (group_size * len(prompts)): group-normalized rewards for each rollout response
        raw_rewards (group_size * len(prompts)): unnormalized rewards
        metadata: other metadata statistics (eg. max / min etc of group rewards)
    """
    rewards = torch.tensor(
        [
        reward_fn(rollout, ground_truth)['reward']
        for rollout, ground_truth in zip(rollout_responses, repeated_ground_truths)
        ],
        dtype=torch.float
    )  # rollout_batch_size,
    
    rewards = rewards.view(-1, group_size)  # num_prompts x group_size
    group_mean = torch.mean(rewards, dim=-1, keepdim=True)
    group_std = torch.std(rewards, dim=-1, keepdim=True)
    metadata = {
        "group_mean": group_mean,
        "group_std": group_std,
        "group_min":  torch.min(rewards, dim=-1)[0],
        "group_max": torch.max(rewards, dim=-1)[0],
        "all_sum": torch.sum(rewards),
        "all_mean": torch.mean(rewards).item(),
        "all_min": torch.min(rewards).item(),
        "all_max": torch.max(rewards).item(),
    }
    
    advantages = (rewards - group_mean)
    if normalize_by_std:
        advantages = advantages / (group_std + advantage_eps)  # prompts x group_size 
    
    return advantages.view(-1), rewards.view(-1), metadata

def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    """Computes -A_t x log(p_\theta(o_t | q, o <t)).
    
    Args:
        - raw_rewards_or_advantages (torch.Tensor): rewards or advantages of shape (batch_size, 1)
        - policy_log_probs (torch.Tensor): log probs of policy of shape (batch_size, seq_length)
    
    Returns:
        - naive_policy_grad_loss (torch.Tensor): of shape batch_size x seq_length
    """
    return -raw_rewards_or_advantages * policy_log_probs

def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute per token grpo clip loss.
    
    Args:
        - advantages (torch.Tensor): per example advantages of shape (batch_size, 1)
        - policy_log_probs (torch.Tensor): per token log probs of shape (batch_size, seq_len)
        - old_log_probs (torch.Tensor): per token old log probs of shape (batch_size, seq_len)
        - cliprange (float): clip param ϵ
    
    Returns:
        - loss (torch.Tensor): per token clipped loss of shape (batch_size, seq_length)
        - metadata (dict): logging things (eg. which token was clipped vs not)
    """
    policy_grad_ratio = torch.exp(policy_log_probs - old_log_probs)  # bsz x seq
    clipped = torch.clamp(policy_grad_ratio, min=1 - cliprange, max=1 + cliprange) * advantages
    
    # which token was clipped
    clipped_mask = torch.zeros_like(clipped)
    clipped_mask[policy_grad_ratio > 1 + cliprange] = 1
    clipped_mask[policy_grad_ratio < 1 - cliprange] = 1

    policy_grad_ratio = policy_grad_ratio * advantages
    
    grpo_per_token_loss = -torch.minimum(
        policy_grad_ratio,
        clipped,
    )

    return grpo_per_token_loss, {"clipped_mask": clipped_mask}

def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None= None,
    advantages: torch.Tensor | None= None,
    old_log_probs: torch.Tensor | None= None,
    cliprange: float | None= None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Dispatch the right loss based on `loss_type` and returns per-token loss and statistics.
    
    Args:
        - policy_log_probs (torch.Tensor): log probs obtained from current policy (bsz x seq)
        - loss_type (literal_type): Can take the following 3 values -
            `no_baseline`: raw rewards as advantages
            `reinforce_with_baseline`: naive policy gradient loss using group-normalized rewards
            `grpo_clip`: GRPO clip loss
        - raw_rewards (torch.Tensor): raw rewards (bsz x 1)
        - advantages (torch.Tensor): advantages (bsz x 1)
        - old_log_probs (torch.Tensor): old policy log probs (bsz x seq)
        - cliprange (float): Used for grpo clip loss
    
    Return:
        - loss (torch.Tensor): per token loss (bsz x seq)
        - metadata (dict): other metadata fields
    """
    
    if loss_type == "no_baseline":
        assert raw_rewards is not None, f"`raw_rewards` cannot be None for loss_type={loss_type}"
        return compute_naive_policy_gradient_loss(
            raw_rewards_or_advantages=raw_rewards,
            policy_log_probs=policy_log_probs
        ), {}
    elif loss_type == "reinforce_with_baseline":
        assert advantages is not None, "advantages cannot be None"
        return compute_naive_policy_gradient_loss(
            raw_rewards_or_advantages=advantages, policy_log_probs=policy_log_probs
        ), {}
    elif loss_type == "grpo_clip":
        assert advantages is not None and old_log_probs is not None and cliprange is not None, \
            "advantages, old_log_probs and cliprange cannot be None"
        return compute_grpo_clip_loss(
            advantages=advantages, policy_log_probs=policy_log_probs, old_log_probs=old_log_probs, cliprange=cliprange
        )
    else:
        raise NotImplementedError(f"`compute_policy_gradient_loss` not implemented for `loss_type`={loss_type}")

def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None
) -> torch.Tensor:
    """Average tensor elements while respecting mask.
    
    Args:
        tensor (torch.Tensor): tensor to average on
        mask (torch.Tensor): mask elements showing which indices to average on
        dim (int | None): dim to average on
    """
    norm_constant = torch.sum(mask, dim=dim)
    return masked_normalize(tensor=tensor, mask=mask, normalize_constant=norm_constant, dim=dim)

def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None= None,
    advantages: torch.Tensor | None= None,
    old_log_probs: torch.Tensor | None= None,
    cliprange: float | None= None,
    length_normalize: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    
    loss, metadata = compute_policy_gradient_loss(
        policy_log_probs=policy_log_probs,
        loss_type=loss_type,
        raw_rewards=raw_rewards,
        advantages=advantages,
        old_log_probs=old_log_probs,
        cliprange=cliprange
    )  # bsz x seq
    
    if length_normalize:
        max_len = torch.max(torch.sum(response_mask, dim=-1))
        loss = masked_normalize(loss, response_mask, normalize_constant=max_len)
    else:
        loss = masked_mean(loss, response_mask) / gradient_accumulation_steps
    loss.backward()
    return loss, metadata