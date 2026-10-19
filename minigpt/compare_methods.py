"""Fixed-budget GPT-2 adapter comparison. Run each method/seed in a fresh process."""
import argparse
import gc
import hashlib
import json
import platform
import random
import time
from pathlib import Path
import torch
import tiktoken
from .adapters import attach, trainable_state, load_trainable
from .data import read_rows, encode_rows, loader, prompt
from .model import GPT, Config
from .train import evaluate, loss_sum


def write(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')


def seed_all(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def build(checkpoint, method, seed, rank=8):
    seed_all(seed)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model = GPT(Config(**saved['config']))
    model.load_state_dict(saved['model'])
    del saved
    model = attach(model, method, rank)
    return model.to('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--init', required=True)
    p.add_argument('--data', default='data/instructions')
    p.add_argument('--out', required=True)
    p.add_argument('--method', choices=['full', 'lora', 'rslora', 'dora', 'qlora'], required=True)
    p.add_argument('--seed', type=int, default=123)
    p.add_argument('--rank', type=int, default=8)
    p.add_argument('--epochs', type=int, default=2)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--lr', type=float)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = build(args.init, args.method, args.seed, args.rank)
    tok = tiktoken.get_encoding('gpt2')
    paths = [Path(args.data) / f'{split}.jsonl' for split in ('train', 'val', 'test')]
    rows = [read_rows(path) for path in paths]
    keys = [set(prompt(row) for row in split) for split in rows]
    assert all(not keys[i] & keys[j] for i, j in ((0, 1), (0, 2), (1, 2))), 'Split leakage'
    batches = [loader(encode_rows(r, tok, 'sft', model.cfg.context_length), tok.eot_token,
                      args.batch_size, shuffle=i == 0) for i, r in enumerate(rows)]
    # Adapter initialization consumes RNG; isolate data order for fair comparisons.
    seed_all(args.seed + 10000)
    lr = args.lr or (5e-5 if args.method == 'full' else 5e-4)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr)
    initial = evaluate(model, batches[1], device)
    meta = dict(vars(args), lr_used=lr, initial_validation=initial,
                torch=torch.__version__, python=platform.python_version(),
                device=torch.cuda.get_device_name() if device == 'cuda' else device,
                trainable_parameters=sum(p.numel() for p in params),
                data_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                precision='FP32 compute; NF4 transformer Linear weights only for qlora',
                quantized_modules=48 if args.method == 'qlora' else 0)
    write(out / 'metadata.json', meta)
    print('READY ' + json.dumps(dict(method=args.method, seed=args.seed,
          initial_validation=initial, trainable_parameters=meta['trainable_parameters'])), flush=True)
    best = float('inf')
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        synchronize()
        if device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        start, total, count = time.perf_counter(), 0., 0
        for i, batch in enumerate(batches[0]):
            optimizer.zero_grad(set_to_none=True)
            loss, n, _, _ = loss_sum(model, batch, device, False)
            (loss / n).backward()
            torch.nn.utils.clip_grad_norm_(params, 1.)
            optimizer.step()
            total += loss.item()
            count += n
            if (i + 1) % 50 == 0:
                print('PROGRESS ' + json.dumps(dict(epoch=epoch, batch=i + 1,
                      seconds=time.perf_counter() - start)), flush=True)
        synchronize()
        elapsed = time.perf_counter() - start
        peak = torch.cuda.max_memory_allocated() if device == 'cuda' else None
        val = evaluate(model, batches[1], device)
        row = dict(epoch=epoch, train_loss=total / count, validation=val,
                   training_seconds=elapsed, peak_allocated_bytes=peak, updates=len(batches[0]))
        history.append(row)
        print(json.dumps(dict(method=args.method, seed=args.seed, **row)), flush=True)
        write(out / 'history.json', history)
        if val['loss'] < best:
            best = val['loss']
            torch.save(trainable_state(model), out / 'selected-trainable.pt')
            best_epoch = epoch
    # Free optimizer state before loading best full-tuning parameters on CPU.
    del optimizer
    gc.collect()
    load_trainable(model, torch.load(out / 'selected-trainable.pt', map_location='cpu', weights_only=True))
    test = evaluate(model, batches[2], device)
    from .generate import generate
    samples = []
    model.eval()
    for row in rows[2][:5]:
        ids = torch.tensor([tok.encode(prompt(row))], device=device)
        with torch.no_grad():
            result = generate(model, ids, 64, eos_id=tok.eot_token)
        samples.append(dict(row, generated=tok.decode(result[0, ids.shape[1]:].tolist())))
    write(out / 'generations.json', samples)
    result = dict(method=args.method, seed=args.seed, selected_epoch=best_epoch,
                  validation_ppl=__import__('math').exp(best), test=test,
                  trainable_parameters=meta['trainable_parameters'],
                  training_seconds=sum(r['training_seconds'] for r in history),
                  peak_allocated_bytes=max((r['peak_allocated_bytes'] or 0) for r in history))
    write(out / 'result.json', result)
    print('RESULT ' + json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
