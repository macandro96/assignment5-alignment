from collections import deque

import pandas as pd
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from cs336_alignment.utils import tokenize_prompt_and_output


class SFTDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        prompt_col: str = "prompt",
        response_col: str = "response",
        sample_rows: int = 0,
    ):
        data = pd.read_json(data_path, lines=True)
        if sample_rows > 0:
            data = data.sample(sample_rows)

        self.prompts = data[prompt_col].tolist()
        self.responses = data[response_col].tolist()

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, idx: int) -> tuple[str, str]:
        return self.prompts[idx], self.responses[idx]


def sft_dataloader(
    data_path: str,
    batch_size: int,
    num_workers: int,
    tokenizer: AutoTokenizer,
    sample_rows: int = 0,
    prompt_col: str = "prompt",
    response_col: str = "response",
) -> DataLoader:
    def collate_fn(batch):
        prompts = [x[0] for x in batch]
        responses = [x[1] for x in batch]
        return tokenize_prompt_and_output(
            prompt_strs=prompts, output_strs=responses, tokenizer=tokenizer
        )

    dataset = SFTDataset(
        data_path=data_path,
        prompt_col=prompt_col,
        response_col=response_col,
        sample_rows=sample_rows,
    )
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )
    return dataloader

class ReplayBuffer:
    def __init__(self):
        self.buffer = deque()
    
    def push(self, new_entries: list):
        self.buffer.extend(new_entries)
    
    def pop(self, batch_size):
        limit = min(len(self.buffer), batch_size)
        items = []
        for _ in range(limit):
            items.append(self.buffer.popleft())
        return items

    def __len__(self):
        return len(self.buffer)

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-1.5B")
    dl = sft_dataloader(
        data_path="/scratch/am10150/projects/cs336/assignment5-alignment/data/math/sft_clean.jsonl",
        batch_size=4,
        num_workers=1,
        tokenizer=tokenizer,
    )
    it = iter(dl)
    batch = next(it)
    print(batch.keys())
    for key, val in batch.items():
        print(f"Key: {key}\ttensor shape: {val.shape}")
    import pdb; pdb.set_trace()
    import pdb; pdb.set_trace()
