"""Generate from an experiment adapter (only load your own trusted checkpoints)."""
import argparse
import torch
import tiktoken
from .adapters import load_trainable
from .compare_methods import build
from .data import prompt
from .generate import generate


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base', required=True)
    p.add_argument('--adapter', required=True)
    p.add_argument('--method', choices=['lora', 'rslora', 'dora', 'qlora'], required=True)
    p.add_argument('--rank', type=int, default=8)
    p.add_argument('--instruction', required=True)
    p.add_argument('--max-new-tokens', type=int, default=64)
    args = p.parse_args()
    model = build(args.base, args.method, 123, args.rank)
    load_trainable(model, torch.load(args.adapter, map_location='cpu', weights_only=True))
    tok = tiktoken.get_encoding('gpt2')
    ids = torch.tensor([tok.encode(prompt({'instruction': args.instruction}))], device=next(model.parameters()).device)
    result = generate(model, ids, args.max_new_tokens, eos_id=tok.eot_token)
    print(tok.decode(result[0, ids.shape[1]:].tolist()))


if __name__ == '__main__':
    main()
