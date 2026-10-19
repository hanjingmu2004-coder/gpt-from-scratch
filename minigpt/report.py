"""Plot recorded training curves without re-evaluating the test set."""
import argparse
import json
import os
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True)
    args = p.parse_args()
    root = Path(args.run)
    rows = [json.loads(line) for line in (root / 'metrics.jsonl').read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError('No completed epochs')
    os.environ.setdefault('MPLCONFIGDIR', str((root / '.matplotlib').resolve()))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([r['epoch'] for r in rows], [r['train_loss'] for r in rows], 'o-', label='Train')
    ax.plot([r['epoch'] for r in rows], [r['validation']['loss'] for r in rows], 'o-', label='Validation')
    ax.set(xlabel='Epoch', ylabel='Cross-entropy', title='MiniGPT learning curve')
    ax.legend()
    fig.tight_layout()
    fig.savefig(root / 'learning_curve.png', dpi=160)
    plt.close(fig)
    summary = {'completed_epochs': len(rows), 'best_validation_loss': min(r['validation']['loss'] for r in rows),
               'epoch_seconds_sum': sum(r['seconds'] for r in rows),
               'test': json.loads((root / 'test_metrics.json').read_text())}
    (root / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
