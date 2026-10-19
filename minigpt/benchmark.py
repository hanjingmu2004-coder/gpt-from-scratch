"""Measure generation including prefill; never report this as decode-only throughput."""
import argparse
import json
import statistics
import time
import torch
from .model import Config, GPT
from .generate import generate


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--input-length', type=int, default=32)
    p.add_argument('--new-tokens', type=int, default=64)
    p.add_argument('--repeats', type=int, default=5)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()
    if min(args.input_length, args.new_tokens, args.repeats) < 1:
        p.error('Lengths and repeats must be positive')
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model = GPT(Config(**state['config'])).to(args.device).eval()
    model.load_state_dict(state['model'])
    torch.manual_seed(123)
    ids = torch.randint(model.cfg.vocab_size, (1, args.input_length), device=args.device)
    if args.input_length > model.cfg.context_length:
        p.error('Input exceeds context')
    def sync():
        if args.device.startswith('cuda'):
            torch.cuda.synchronize()
    cached = generate(model, ids, args.new_tokens)
    plain = generate(model, ids, args.new_tokens, use_cache=False)
    if not torch.equal(cached, plain):
        raise RuntimeError('Greedy token outputs differ; inspect numerical tolerance/logits before benchmarking')
    results = dict(vars(args), hardware=torch.cuda.get_device_name() if args.device.startswith('cuda') else 'CPU')
    for cache in (False, True):
        generate(model, ids, args.new_tokens, use_cache=cache)
        sync()
        if args.device.startswith('cuda'):
            torch.cuda.reset_peak_memory_stats()
        durations, first = [], []
        for _ in range(args.repeats):
            sync()
            begin = time.perf_counter()
            generate(model, ids, 1, use_cache=cache)
            sync()
            first.append(time.perf_counter() - begin)
            begin = time.perf_counter()
            generate(model, ids, args.new_tokens, use_cache=cache)
            sync()
            durations.append(time.perf_counter() - begin)
        elapsed = statistics.median(durations)
        results['cache' if cache else 'no_cache'] = dict(
            median_total_seconds=elapsed, tokens_per_second_including_prefill=args.new_tokens / elapsed,
            median_first_token_seconds=statistics.median(first),
            peak_allocated_bytes=torch.cuda.max_memory_allocated() if args.device.startswith('cuda') else None)
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
