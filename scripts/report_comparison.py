"""Aggregate saved real results without inventing missing runs."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics
import os
import sys


def stats(values):
    return dict(mean=statistics.mean(values), std=statistics.stdev(values) if len(values) > 1 else None,
                n=len(values), values=values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run')
    args = parser.parse_args()
    run = Path(args.run)
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    sft, align = defaultdict(list), defaultdict(list)
    for path in sorted((run / 'sft').glob('*/result.json')):
        row = json.loads(path.read_text())
        sft[row['method']].append(row)
    for path in sorted((run / 'preferences').glob('*/COMPLETE.json')):
        for row in json.loads((path.parent / 'results.json').read_text()):
            align[row['method']].append(row)
    # Fail on duplicate seeds instead of silently treating retries as repeats.
    for group in (sft, align):
        for method, rows in group.items():
            seeds = [r['seed'] for r in rows]
            if len(seeds) != len(set(seeds)):
                raise ValueError(f'Duplicate completed seeds for {method}: {seeds}')
    summary = dict(sft={}, preferences={})
    lines = ['# GPT-2 方法对照实验：实际结果', '',
             '均值 ± 样本标准差；三个种子是探索性重复，不构成通用排名。', '',
             '## 指令微调', '',
             '| 方法 | n | 测试回答 PPL ↓ | 参数数 | GPU 峰值 GiB ↓ | 两轮训练秒 ↓ |',
             '|---|---:|---:|---:|---:|---:|']
    def formatted(values, precision=2):
        s = stats(values)
        return f'{s["mean"]:.{precision}f} ± {s["std"]:.{precision}f}' if s['std'] is not None else f'{s["mean"]:.{precision}f} (单次)'
    order = ['full', 'lora', 'rslora', 'dora', 'qlora']
    for method in order:
        rows = sft.get(method, [])
        if not rows:
            continue
        metrics = dict(test_ppl=[r['test']['perplexity'] for r in rows],
                       test_nll=[r['test']['loss'] for r in rows],
                       validation_ppl=[r['validation_ppl'] for r in rows],
                       gpu_gib=[r['peak_allocated_bytes'] / 2**30 for r in rows],
                       seconds=[r['training_seconds'] for r in rows])
        summary['sft'][method] = dict(seeds=[r['seed'] for r in rows],
                    trainable_parameters=rows[0]['trainable_parameters'],
                    metrics={k: stats(v) for k, v in metrics.items()})
        lines.append(f'| {method} | {len(rows)} | {formatted(metrics["test_ppl"])} | {rows[0]["trainable_parameters"]:,} | '
                     f'{formatted(metrics["gpu_gib"])} | {formatted(metrics["seconds"])} |')
    lines += ['', '## 二标签单步偏好/奖励实验', '',
              'SST-2 512/128/256；不是官方 GLUE 测试。受约束的两个动作，不是长文本生成。', '',
              '测试集 negative=122、positive=134；恒预测 positive 的准确率为 52.34%。', '',
              '| 方法 | n | 测试准确率 % ↑ | 测试 binary NLL ↓ | 选择步数 |',
              '|---|---:|---:|---:|---|']
    for method in ('warm_sft', 'sft_continued', 'dpo', 'ppo', 'grpo'):
        rows = align.get(method, [])
        if not rows:
            continue
        accuracy = [r['test']['accuracy'] * 100 for r in rows]
        nll = [r['test']['binary_nll'] for r in rows]
        summary['preferences'][method] = dict(seeds=[r['seed'] for r in rows],
                 accuracy_percent=stats(accuracy), binary_nll=stats(nll),
                 selected_steps=[r.get('selected_step') for r in rows])
        lines.append(f'| {method} | {len(rows)} | {formatted(accuracy)} | {formatted(nll, 3)} | '
                     + ', '.join(str(r.get('selected_step', 'warm')) for r in rows) + ' |')
    if 'warm_sft' in align:
        baseline = {r['seed']: r for r in align['warm_sft']}
        for method, rows in align.items():
            if method == 'warm_sft':
                continue
            deltas = [(r['test']['accuracy'] - baseline[r['seed']]['test']['accuracy']) * 100 for r in rows]
            summary['preferences'][method]['paired_accuracy_delta_pp'] = stats(deltas)
    lines += ['', '## 解释边界', '',
              '- SFT full LR=5e-5，adapter LR=5e-4；rank=8、alpha=16，均未搜索超参数。',
              '- QLoRA 是 NF4 Transformer Linear + double quantization、FP32 计算，未量化 embedding/共享 head，未启用 paged optimizer。',
              '- DoRA 显式构造有效权重，资源开销受本实现影响，不能代表优化后的 PEFT 实现。',
              '- 指令 SFT 按验证 NLL 选模型；奖励实验按验证准确率选取，NLL 用于平局。PPL 不等于事实正确率。',
              '- 奖励实验来自真实标签，偏好对自动构造；PPO critic 是共享骨干的标量头。',
              '- 选中第 0 步表示验证集选择保留 warm start；不代表训练带来提升。',
              '- PPO/GRPO 有采样成本；虽然更新次数相同，信息和计算预算并不完全等价。',
              '- 完整定义、数据指纹、实现与限制见 docs/METHOD_COMPARISON.md。', '']
    (run / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    (run / 'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
    # Independent hashes make downloaded evidence verifiable.
    hashes = {str(p.relative_to(run)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in run.rglob('*.json') if p.name != 'evidence-sha256.json'}
    (run / 'evidence-sha256.json').write_text(json.dumps(hashes, indent=2), encoding='utf-8')
    cache = run / '.matplotlib'
    cache.mkdir(exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', str(cache.resolve()))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    methods = [m for m in order if m in sft]
    for ax, key, title in ((axes[0], 'test_ppl', 'Instruction test answer PPL (lower is better)'),
                           (axes[1], 'gpu_gib', 'SFT peak GPU allocated (GiB)')):
        means = [summary['sft'][m]['metrics'][key]['mean'] for m in methods]
        errors = [summary['sft'][m]['metrics'][key]['std'] or 0 for m in methods]
        ax.bar(methods, means, yerr=errors, capsize=4)
        ax.set_title(title, fontsize=10)
    methods = list(summary['preferences'])
    axes[2].bar(methods, [summary['preferences'][m]['accuracy_percent']['mean'] for m in methods],
                yerr=[summary['preferences'][m]['accuracy_percent']['std'] or 0 for m in methods], capsize=4)
    axes[2].set_title('Binary SST-2 test accuracy (%)', fontsize=10)
    axes[2].set_ylim(0, 100)
    for ax in axes:
        ax.tick_params(axis='x', rotation=25)
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
    fig.suptitle('GPT-2 124M: fixed-configuration controls; error bars = sample SD across seeds')
    fig.tight_layout()
    fig.savefig(run / 'comparison.png', dpi=170)
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
