"""Merge a native FP32 adapter into its original pretrained base on CPU."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from .adapters import LowRank, attach, load_trainable, trainable_state
from .model import Config, GPT


@torch.no_grad()
def merge(model):
    adapters = [(name, module) for name, module in model.named_modules()
                if isinstance(module, LowRank)]
    if any(module.method == 'qlora' for _, module in adapters):
        raise ValueError('Quantized adapters require a separate dequantization workflow')
    for name, module in adapters:
        weight = module.base.weight + module.scale * (module.b @ module.a)
        if module.method == 'dora':
            weight *= (module.magnitude / weight.norm(dim=1).clamp_min(1e-8))[:, None]
        module.base.weight.copy_(weight)
        parent_name, child = name.rsplit('.', 1)
        setattr(model.get_submodule(parent_name), child, module.base)
    return model


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True, help='Original, unfine-tuned native GPT checkpoint')
    parser.add_argument('--adapter', required=True)
    parser.add_argument('--method', required=True, choices=['lora', 'rslora', 'dora'])
    parser.add_argument('--rank', type=int, default=8)
    parser.add_argument('--alpha', type=float, default=16.)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists():
        raise ValueError('Output already exists; choose a new path')
    checkpoint = torch.load(args.base, map_location='cpu', weights_only=True)
    if checkpoint.get('task') != 'lm':
        raise ValueError('Use the original pretrained LM base, not a fine-tuned checkpoint')
    model = GPT(Config(**checkpoint['config'])).eval()
    model.load_state_dict(checkpoint['model'], strict=True)
    attach(model, args.method, args.rank, args.alpha)
    adapter = torch.load(args.adapter, map_location='cpu', weights_only=True)
    expected = trainable_state(model)
    if set(adapter) != set(expected):
        raise ValueError('Adapter parameter keys do not match the model')
    for name, tensor in adapter.items():
        if tensor.shape != expected[name].shape or not torch.isfinite(tensor).all():
            raise ValueError(f'Invalid adapter tensor: {name}')
    load_trainable(model, adapter)
    torch.manual_seed(123)
    ids = [torch.randint(checkpoint['config']['vocab_size'], (1, length))
           for length in (8, min(31, checkpoint['config']['context_length']))]
    with torch.no_grad():
        before = [model(tokens).clone() for tokens in ids]
        merge(model)
        errors = []
        for tokens, reference in zip(ids, before):
            actual = model(tokens)
            torch.testing.assert_close(actual, reference, atol=2e-4, rtol=2e-4)
            errors.append(float((actual - reference).abs().max()))
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(task='sft', config=checkpoint['config'], model=model.state_dict()), out)
    record = dict(base_sha256=sha256(args.base), adapter_sha256=sha256(args.adapter),
                  model_sha256=sha256(out), method=args.method, rank=args.rank,
                  alpha=args.alpha, max_abs_errors=errors, bytes=out.stat().st_size)
    out.with_suffix('.merge.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
