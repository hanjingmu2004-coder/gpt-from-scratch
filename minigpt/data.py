import json
from functools import partial
from pathlib import Path
import torch
from torch.utils.data import DataLoader


def prompt(row):
    return (f"### Instruction:\n{row['instruction']}\n\n"
            + (f"### Input:\n{row['input']}\n\n" if row.get('input') else '')
            + "### Response:\n")


def read_rows(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"Empty dataset: {path}")
    return rows


def encode_rows(rows, tokenizer, task, context):
    samples = []
    for row in rows:
        if task == "classification":
            ids = tokenizer.encode(row['text'])[:context]
            if not ids:
                raise ValueError("Classification text must not be empty")
            samples.append((ids, int(row['label'])))
        elif task == "sft":
            # Encode separately so the prompt/response boundary is unambiguous.
            prefix = tokenizer.encode(prompt(row))
            answer = tokenizer.encode(row['output']) + [tokenizer.eot_token]
            tokens = (prefix + answer)[:context + 1]
            x, y = tokens[:-1], tokens[1:]
            y = [-100 if i < len(prefix) - 1 else target for i, target in enumerate(y)]
            if not any(t != -100 for t in y):
                raise ValueError("Prompt fills the context; shorten prompt or increase context")
            samples.append((x, y))
        else:
            tokens = tokenizer.encode(row['text']) + [tokenizer.eot_token]
            for start in range(0, len(tokens) - 1, context):
                chunk = tokens[start:start + context + 1]
                samples.append((chunk[:-1], chunk[1:]))
    if not samples:
        raise ValueError("No usable samples")
    return samples


def collate(batch, pad, classification=False):
    lengths = torch.tensor([len(x) for x, _ in batch])
    width = int(lengths.max())
    ids = torch.full((len(batch), width), pad, dtype=torch.long)
    labels = (torch.tensor([y for _, y in batch]) if classification else
              torch.full_like(ids, -100))
    for i, (x, y) in enumerate(batch):
        ids[i, :len(x)] = torch.tensor(x)
        if not classification:
            labels[i, :len(y)] = torch.tensor(y)
    return ids, labels, lengths


def loader(samples, pad, batch_size, classification=False, shuffle=False):
    return DataLoader(samples, batch_size=batch_size, shuffle=shuffle,
                      collate_fn=partial(collate, pad=pad, classification=classification))
