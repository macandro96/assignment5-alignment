import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)


def _get_dtype(dtype: str) -> torch.dtype:
    if dtype == "bf16":
        return torch.bfloat16
    elif dtype == "fp16":
        return torch.float16
    elif dtype == "fp32":
        return torch.float32


def load_model(model_cfg: dict, device: str | None = None):
    """Load model and tokenizer according to config.

    This loads the model using HuggingFace `from_pretrained` and moves it to
    `device` if provided. For large models you may want to switch to a
    device-mapping strategy outside this helper.
    """
    dtype = _get_dtype(model_cfg.get("dtype", "fp32"))
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["name"],
        torch_dtype=dtype,
        attn_implementation="flash_attention_2",
        device_map=device,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["name"])

    return model, tokenizer


def load_optimizer_and_scheduler(
    optim_cfg: dict,
    params,
    total_training_steps: int | None = None
) -> dict:
    if optim_cfg["type"]["name"] == "AdamW":
        optimizer = torch.optim.AdamW(
            params=params,
            lr=float(optim_cfg["lr"]),
            betas=(optim_cfg["type"]["beta1"], optim_cfg["type"]["beta2"]),
        )
    else:
        raise NotImplementedError(f"{optim_cfg['type']['name']} not implemented")

    scheduler = None
    if "scheduler" in optim_cfg and total_training_steps:
        warmup_ratio = optim_cfg["scheduler"]["warmup_ratio"]
        scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=int(warmup_ratio * total_training_steps),
            num_training_steps=total_training_steps,
        )
    return {"optimizer": optimizer, "scheduler": scheduler}

def save_model_tokenizer(model, tokenizer, save_dir: str):
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
