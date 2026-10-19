"""GPT-2 + LoRA, binary single-token SST-2 contextual-bandit experiment.

The action space is explicitly restricted to two verbalizers. This tests objective
behavior and sentiment generalization, NOT free-generation reasoning or full RLHF.
PPO uses terminal reward minus a learned value, two clipped policy updates and MSE
critic. GRPO uses four sampled actions per prompt and a normalized group baseline.
Both regularize with exact categorical KL to the shared warm-SFT reference.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import random
import time
import torch
from torch import nn
from torch.nn import functional as F
import tiktoken
from .adapters import trainable_state, load_trainable
from .compare_methods import build, seed_all, synchronize, write
from .data import read_rows
from .preference_objectives import dpo, exact_kl, group_advantages, clipped_policy_loss


def tensor_batch(rows, indices, tokenizer, device):
    texts = [f'Review: {rows[i]["sentence"].strip()}\nSentiment:' for i in indices]
    tokens = [tokenizer.encode(text)[:128] for text in texts]
    lengths = torch.tensor([len(t) for t in tokens], device=device)
    ids = torch.full((len(tokens), int(lengths.max())), tokenizer.eot_token, device=device, dtype=torch.long)
    for i, ts in enumerate(tokens):
        ids[i, :len(ts)] = torch.tensor(ts, device=device)
    labels = torch.tensor([rows[i]['label'] for i in indices], device=device)
    return ids, lengths, labels


def policy(model, batch, verbalizers, value=None):
    ids, lengths, _ = batch
    hidden, _ = model.hidden(ids)
    last = hidden[torch.arange(ids.size(0), device=ids.device), lengths - 1]
    scores = F.linear(last, model.head.weight[verbalizers])
    return scores.log_softmax(-1), value(last).squeeze(-1) if value is not None else None


@torch.no_grad()
def evaluate(model, rows, tokenizer, device, verbalizers):
    model.eval()
    total, correct, predictions = 0., 0, []
    for start in range(0, len(rows), 16):
        batch = tensor_batch(rows, list(range(start, min(start + 16, len(rows)))), tokenizer, device)
        logp, _ = policy(model, batch, verbalizers)
        labels = batch[2]
        total += F.nll_loss(logp, labels, reduction='sum').item()
        correct += (logp.argmax(-1) == labels).sum().item()
        predictions.extend(logp.argmax(-1).cpu().tolist())
    return dict(accuracy=correct / len(rows), binary_nll=total / len(rows),
                examples=len(rows), predictions=predictions)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--init', required=True)
    p.add_argument('--data', default='data/sentiment-control')
    p.add_argument('--out', required=True)
    p.add_argument('--seed', type=int, default=123)
    p.add_argument('--steps', type=int, default=40)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tok = tiktoken.get_encoding('gpt2')
    words = [tok.encode(w) for w in (' negative', ' positive')]
    assert all(len(w) == 1 for w in words), words
    verbalizers = [w[0] for w in words]
    rows = {s: read_rows(Path(args.data) / f'{s}.jsonl') for s in ('train', 'val', 'test')}
    sets = [set(' '.join(r['sentence'].lower().split()) for r in rows[s]) for s in ('train', 'val', 'test')]
    assert all(not sets[i] & sets[j] for i, j in ((0, 1), (0, 2), (1, 2))), 'Duplicate split texts'
    write(out / 'metadata.json', dict(vars(args), torch=torch.__version__,
          gpu=torch.cuda.get_device_name() if device == 'cuda' else device,
          verbalizer_ids=verbalizers, rank=8, alpha=16, lr=5e-4, beta=.1, group_size=4,
          warm_epochs=2, batch_size=16, inner_updates=2, clip=.2,
          selection='alignment: validation accuracy, ties by lower NLL; warm SFT: validation NLL',
          action_space='ONLY two sentiment labels; one terminal decision per prompt',
          data_sha256={s: hashlib.sha256((Path(args.data) / f'{s}.jsonl').read_bytes()).hexdigest() for s in rows}))
    # The warm-start, reference and validation selection are shared by all methods.
    model = build(args.init, 'lora', args.seed)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-4)
    best, warm_history = float('inf'), []
    for epoch in range(2):
        order = list(range(len(rows['train'])))
        random.Random(args.seed + epoch).shuffle(order)
        model.train()
        for start in range(0, len(order), 16):
            batch = tensor_batch(rows['train'], order[start:start + 16], tok, device)
            optimizer.zero_grad(set_to_none=True)
            logp, _ = policy(model, batch, verbalizers)
            F.nll_loss(logp, batch[2]).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
        val = evaluate(model, rows['val'], tok, device, verbalizers)
        warm_history.append(dict(epoch=epoch + 1, **val))
        if val['binary_nll'] < best:
            best = val['binary_nll']
            warm = trainable_state(model)
        print('WARM ' + json.dumps(dict(seed=args.seed, epoch=epoch + 1, val_accuracy=val['accuracy'])), flush=True)
    load_trainable(model, warm)
    write(out / 'warm_history.json', warm_history)
    torch.save(warm, out / 'warm-adapter.pt')
    # Cache the reference distribution before any policy optimization.
    reference = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(rows['train']), 16):
            batch = tensor_batch(rows['train'], list(range(start, min(start + 16, len(rows['train'])))), tok, device)
            logp, _ = policy(model, batch, verbalizers)
            reference.append(logp.cpu())
    reference = torch.cat(reference).to(device)
    baseline_val = evaluate(model, rows['val'], tok, device, verbalizers)
    baseline_test = evaluate(model, rows['test'], tok, device, verbalizers)
    results = [dict(method='warm_sft', seed=args.seed, validation=baseline_val, test=baseline_test)]
    write(out / 'results.json', results)
    del optimizer, model
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()
    for method in ('sft_continued', 'dpo', 'ppo', 'grpo'):
        model = build(args.init, 'lora', args.seed)
        load_trainable(model, warm)
        seed_all(args.seed + 1000)
        value = nn.Linear(model.cfg.emb_dim, 1).to(device) if method == 'ppo' else None
        if value is not None:
            nn.init.zeros_(value.weight)
            nn.init.zeros_(value.bias)
        params = [p for p in model.parameters() if p.requires_grad] + (list(value.parameters()) if value is not None else [])
        optimizer = torch.optim.AdamW(params, lr=5e-4)
        rng = random.Random(args.seed + 100)
        best, selected, selected_step = (baseline_val['accuracy'], -baseline_val['binary_nll']), warm, 0
        history = []
        synchronize()
        if device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        for step in range(1, args.steps + 1):
            indices = rng.sample(range(len(rows['train'])), 16)
            batch = tensor_batch(rows['train'], indices, tok, device)
            ref = reference[indices]
            model.train()
            stats = {}
            if method in ('ppo', 'grpo'):
                with torch.no_grad():
                    old, old_value = policy(model, batch, verbalizers, value)
                    actions = torch.multinomial(old.exp(), 4, replacement=True)
                    old_action_logp = old.gather(-1, actions)
                    rewards = (actions == batch[2][:, None]).float()
                    advantages = (group_advantages(rewards) if method == 'grpo'
                                  else rewards - old_value[:, None])
                    stats = dict(sampled_reward=rewards.mean().item(),
                                 zero_variance_groups=(rewards.std(-1, correction=0) == 0).float().mean().item())
            for inner in range(2):
                optimizer.zero_grad(set_to_none=True)
                logp, values = policy(model, batch, verbalizers, value)
                kl = exact_kl(logp, ref).mean()
                if method == 'dpo':
                    loss = dpo(logp, ref, batch[2], beta=.1)
                elif method == 'sft_continued':
                    loss = F.nll_loss(logp, batch[2])
                else:
                    loss = clipped_policy_loss(logp, old_action_logp, actions, advantages) + .1 * kl
                    if method == 'ppo':
                        loss = loss + .5 * (values[:, None] - rewards).square().mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.)
                optimizer.step()
            if step % 10 == 0 or step == args.steps:
                val = evaluate(model, rows['val'], tok, device, verbalizers)
                history.append(dict(step=step, validation=val, kl=kl.item(), **stats))
                selection_score = (val['accuracy'], -val['binary_nll'])
                if selection_score > best:
                    best, selected, selected_step = selection_score, trainable_state(model), step
                print('ALIGN ' + json.dumps(dict(method=method, seed=args.seed, step=step,
                      val_accuracy=val['accuracy'], val_nll=val['binary_nll'], **stats)), flush=True)
        synchronize()
        elapsed = time.perf_counter() - started
        peak = torch.cuda.max_memory_allocated() if device == 'cuda' else None
        load_trainable(model, selected)
        val = evaluate(model, rows['val'], tok, device, verbalizers)
        test = evaluate(model, rows['test'], tok, device, verbalizers)
        torch.save(selected, out / f'{method}-adapter.pt')
        write(out / f'{method}-history.json', history)
        result = dict(method=method, seed=args.seed, selected_step=selected_step,
                      validation=val, test=test, seconds_including_validation=elapsed,
                      peak_allocated_bytes=peak, optimizer_updates=2 * args.steps,
                      sampled_actions=16 * 4 * args.steps if method in ('ppo', 'grpo') else 0,
                      preference_pairs=16 * args.steps if method == 'dpo' else 0)
        results.append(result)
        write(out / 'results.json', results)
        print('RESULT ' + json.dumps({k: v for k, v in result.items() if k not in ('validation', 'test')}) +
              f' test_accuracy={test["accuracy"]}', flush=True)
        del optimizer, params, value, model
        gc.collect()
        if device == 'cuda':
            torch.cuda.empty_cache()

    write(out / 'COMPLETE.json', {'seed': args.seed, 'methods': [r['method'] for r in results]})


if __name__ == '__main__':
    main()
