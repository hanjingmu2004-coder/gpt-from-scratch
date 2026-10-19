"""Prepare the instruction dataset already used in the original learning notebook."""
import argparse
import hashlib
import json
from pathlib import Path
import random
from .prepare_data import download

REVISION = 'cdbd33e6d57a71e20ea31434841095fac50cecd8'
BASE = f'https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/{REVISION}'
URL = BASE + '/ch07/01_main-chapter-code/instruction-data.json'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', default='data/instructions')
    args = p.parse_args()
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use a fresh output directory')
    raw = Path('data/raw/instructions')
    download(URL, raw / 'instruction-data.json')
    download(BASE + '/LICENSE.txt', raw / 'LICENSE.txt')
    rows = json.loads((raw / 'instruction-data.json').read_text(encoding='utf-8'))
    unique, seen = [], set()
    for row in rows:
        key = (row['instruction'].strip().casefold(), row.get('input', '').strip().casefold())
        if key not in seen:
            unique.append({k: row.get(k, '') for k in ('instruction', 'input', 'output')})
            seen.add(key)
    random.Random(123).shuffle(unique)
    train_end, val_end = int(len(unique)*.85), int(len(unique)*.9)
    splits = dict(train=unique[:train_end], val=unique[train_end:val_end], test=unique[val_end:])
    out.mkdir(parents=True)
    manifest = dict(source=URL, revision=REVISION, seed=123, raw_entries=len(rows),
                    duplicates_removed=len(rows)-len(unique),
                    raw_sha256=hashlib.sha256((raw/'instruction-data.json').read_bytes()).hexdigest(),
                    provenance='Dataset from LLMs-from-scratch used in original GPT.ipynb; see repository license', splits={})
    for split, records in splits.items():
        path = out / f'{split}.jsonl'
        path.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in records), encoding='utf-8')
        manifest['splits'][split] = dict(examples=len(records), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    (out / 'LICENSE.txt').write_bytes((raw/'LICENSE.txt').read_bytes())
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
