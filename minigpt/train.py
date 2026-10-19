"""Run with python -m minigpt.train --help."""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import random
import time
import torch
from torch.nn import functional as F
from .data import read_rows, encode_rows, loader
from .model import Config, GPT, Classifier


def loss_sum(model, batch, device, classification):
    ids, labels, lengths = (t.to(device) for t in batch)
    logits = model(ids, lengths) if classification else model(ids)
    targets = labels if classification else labels.flatten()
    scores = logits if classification else logits.flatten(0, 1)
    count = int((targets != -100).sum())
    return F.cross_entropy(scores, targets, reduction="sum"), count, logits, labels


@torch.no_grad()
def evaluate(model, batches, device, classification=False, num_classes=2):
    was_training = model.training
    model.eval()
    total, count = 0., 0
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    try:
        for batch in batches:
            loss, n, logits, labels = loss_sum(model, batch, device, classification)
            total += loss.item()
            count += n
            if classification:
                index = labels.cpu() * num_classes + logits.argmax(-1).cpu()
                confusion += torch.bincount(index, minlength=num_classes ** 2).reshape(num_classes, num_classes)
        if not count:
            raise ValueError("No evaluation targets")
        result = {"loss": total / count, "targets": count}
        if classification:
            tp = confusion.diag().float()
            denominator = confusion.sum(0) + confusion.sum(1)
            result.update(accuracy=float(tp.sum() / confusion.sum()),
                          macro_f1=float((2 * tp / denominator.clamp_min(1)).mean()),
                          confusion_matrix=confusion.tolist())
        else:
            result['perplexity'] = math.exp(result['loss']) if result['loss'] < 700 else float('inf')
        return result
    finally:
        model.train(was_training)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--task', choices=['lm', 'classification', 'sft'], required=True)
    p.add_argument('--train', required=True)
    p.add_argument('--val', required=True)
    p.add_argument('--test', required=True)
    p.add_argument('--config', default='configs/small.json')
    p.add_argument('--out', required=True)
    p.add_argument('--init', help='Trusted LM checkpoint for fine-tuning')
    p.add_argument('--resume', help='Trusted checkpoint from the same run')
    p.add_argument('--mode', choices=['head', 'last', 'full'], default='full')
    p.add_argument('--num-classes', type=int, default=2)
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--clip', type=float, default=1.0, help='0 disables clipping')
    p.add_argument('--seed', type=int, default=123)
    p.add_argument('--log-every', type=int, default=50)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()
    if args.init and args.resume:
        p.error('--init and --resume are mutually exclusive')
    if args.epochs < 1 or args.batch_size < 1 or args.clip < 0 or args.log_every < 1:
        p.error('Invalid training settings')
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    import tiktoken
    tok = tiktoken.get_encoding('gpt2')
    cfg = Config(**json.loads(Path(args.config).read_text()))
    if cfg.vocab_size != tok.n_vocab:
        raise ValueError('Config vocabulary must match GPT-2 tokenizer')
    rows = [read_rows(path) for path in (args.train, args.val, args.test)]
    # Exact duplicates are rejected across splits, including conflicting labels.
    def key(r):
        content = {k: r.get(k, '') for k in ('instruction', 'input')} if args.task == 'sft' else r['text']
        return json.dumps(content, sort_keys=True, ensure_ascii=False)
    sets = [set(map(key, split)) for split in rows]
    if any(sets[i] & sets[j] for i, j in ((0, 1), (0, 2), (1, 2))):
        raise ValueError('Data leakage: identical content across splits')
    if args.task == 'classification':
        if any(not isinstance(r['label'], int) or not 0 <= r['label'] < args.num_classes for split in rows for r in split):
            raise ValueError('Labels must be integers in [0, num_classes)')
    samples = [encode_rows(split, tok, args.task, cfg.context_length) for split in rows]
    classification = args.task == 'classification'
    batches = [loader(s, tok.eot_token, args.batch_size, classification, shuffle=i == 0)
               for i, s in enumerate(samples)]
    backbone = GPT(cfg)
    if args.init:
        initial = torch.load(args.init, map_location='cpu', weights_only=False)
        if initial['task'] == 'classification' or asdict(Config(**initial['config'])) != asdict(cfg):
            raise ValueError('Initialization requires an LM/SFT checkpoint with matching config')
        backbone.load_state_dict(initial['model'])
    model = Classifier(backbone, args.num_classes) if classification else backbone
    if classification:
        model.configure(args.mode)
    model.to(args.device)
    optimizer = torch.optim.AdamW([v for v in model.parameters() if v.requires_grad], lr=args.lr)
    start, step, best = 0, 0, float('inf')
    hashes = {name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
              for name, path in zip(('train', 'val', 'test'), (args.train, args.val, args.test))}
    signature = {'task': args.task, 'mode': args.mode, 'num_classes': args.num_classes,
                 'config': asdict(cfg), 'data': hashes, 'batch_size': args.batch_size,
                 'lr': args.lr, 'clip': args.clip, 'seed': args.seed}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.resume:
        saved = torch.load(args.resume, map_location='cpu', weights_only=False)
        saved_signature = dict(saved['signature'], config=asdict(Config(**saved['config'])))
        if saved_signature != signature:
            raise ValueError('Resume configuration/data differ from saved run')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        start, step, best = saved['epoch'], saved['step'], saved['best']
        torch.set_rng_state(saved['rng'])
        random.setstate(saved['python_rng'])
        if torch.cuda.is_available() and saved['cuda_rng'] is not None:
            torch.cuda.set_rng_state_all(saved['cuda_rng'])
    elif (out / 'last.pt').exists():
        raise ValueError('Output already contains a run; use --resume or a new directory')
    metadata = dict(vars(args), config=asdict(cfg), data_sha256=hashes,
                    torch_version=torch.__version__, parameters=sum(v.numel() for v in model.parameters()),
                    trainable_parameters=sum(v.numel() for v in model.parameters() if v.requires_grad),
                    hardware=torch.cuda.get_device_name() if args.device.startswith('cuda') else 'CPU')
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(json.dumps({'event': 'ready', 'device': args.device,
                      'parameters': metadata['parameters'],
                      'samples': [len(s) for s in samples],
                      'batches_per_epoch': len(batches[0])}), flush=True)
    if start == 0:
        baseline = evaluate(model, batches[1], args.device, classification, args.num_classes)
        (out / 'initial_validation.json').write_text(json.dumps(baseline, indent=2), encoding='utf-8')
        print('Initial validation:', baseline, flush=True)
    if args.device.startswith('cuda'):
        torch.cuda.reset_peak_memory_stats()
    for epoch in range(start, args.epochs):
        begin = time.perf_counter()
        model.train()
        total, targets, max_grad = 0., 0, 0.
        for batch in batches[0]:
            optimizer.zero_grad(set_to_none=True)
            summed, n, _, _ = loss_sum(model, batch, args.device, classification)
            (summed / n).backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip or float('inf'), error_if_nonfinite=True)
            max_grad = max(max_grad, float(norm))
            optimizer.step()
            total += summed.item()
            targets += n
            step += 1
            if step % args.log_every == 0:
                print(json.dumps({'event': 'progress', 'epoch': epoch + 1, 'step': step,
                                  'running_train_loss': total / targets,
                                  'epoch_seconds': time.perf_counter() - begin}), flush=True)
        validation = evaluate(model, batches[1], args.device, classification, args.num_classes)
        improved = validation['loss'] < best
        best = min(best, validation['loss'])
        record = dict(epoch=epoch + 1, step=step, train_loss=total / targets,
                      validation=validation, seconds=time.perf_counter() - begin,
                      max_grad_norm=max_grad,
                      peak_memory_bytes=torch.cuda.max_memory_allocated() if args.device.startswith('cuda') else None)
        with (out / 'metrics.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(record) + '\n')
        print(json.dumps(record), flush=True)
        state = dict(task=args.task, config=asdict(cfg), signature=signature, model=model.state_dict(),
                     optimizer=optimizer.state_dict(), epoch=epoch + 1, step=step, best=best,
                     rng=torch.get_rng_state(), python_rng=random.getstate(),
                     cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
        def save(name):
            temporary = out / (name + '.tmp')
            torch.save(state, temporary)
            temporary.replace(out / name)
        save('last.pt')
        if improved:
            save('best.pt')
    # Test only the validation-selected checkpoint, never select on test performance.
    best_path = out / 'best.pt'
    if not best_path.exists():
        raise ValueError('Resume in the original output directory containing best.pt')
    model.load_state_dict(torch.load(best_path, map_location='cpu', weights_only=False)['model'])
    result = evaluate(model, batches[2], args.device, classification, args.num_classes)
    (out / 'test_metrics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('Test:', result)


if __name__ == '__main__':
    main()
