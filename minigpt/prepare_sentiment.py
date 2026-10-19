"""Small SST-2 control task; public validation divided into development/test.

Not the official GLUE test score. SST-2 contains train phrases; the heldout
sentences are from its original validation split. No claim of pretraining decontamination.
"""
import hashlib
import json
from pathlib import Path
import random
from huggingface_hub import HfApi, hf_hub_download
import pyarrow.parquet as pq

REVISION = '8d51e7e4887a4caaa95b3fbebbf53c0490b58bbb'


def main():
    out = Path('data/sentiment-control')
    out.mkdir(parents=True, exist_ok=True)
    repo = 'stanfordnlp/sst2'
    revision = REVISION
    names = HfApi().list_repo_files(repo, repo_type='dataset', revision=revision)
    paths = {}
    rows = {}
    for split in ('train', 'validation'):
        name = next(n for n in names if n.startswith(f'data/{split}-') and n.endswith('.parquet'))
        path = hf_hub_download(repo, name, repo_type='dataset', revision=revision)
        paths[split] = {'file': name, 'sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest()}
        rows[split] = pq.read_table(path).to_pylist()
    rng = random.Random(937)
    rng.shuffle(rows['validation'])
    held = rows['validation']
    keys = set(' '.join(r['sentence'].lower().split()) for r in held)
    train, seen = [], set()
    rng.shuffle(rows['train'])
    for row in rows['train']:
        key = ' '.join(row['sentence'].lower().split())
        if key not in keys and key not in seen:
            seen.add(key)
            train.append(row)
        if len(train) == 512:
            break
    splits = {'train': train, 'val': held[:128], 'test': held[128:384]}
    hashes = {}
    for split, records in splits.items():
        path = out / f'{split}.jsonl'
        path.write_text(''.join(json.dumps(r) + '\n' for r in records), encoding='utf-8')
        hashes[split] = hashlib.sha256(path.read_bytes()).hexdigest()
    (out / 'manifest.json').write_text(json.dumps(dict(
        dataset=repo, revision=revision, source_files=paths, sha256=hashes,
        counts={k: len(v) for k, v in splits.items()}, seed=937,
        license='Dataset card: unknown; upstream https://nlp.stanford.edu/sentiment/',
        split_note='512 original train phrases; 128/256 disjoint original-validation sentences; not official GLUE test',
        reward='1 if sampled label equals the original annotation, else 0; preferences automatically derived from labels'), indent=2))
    print((out / 'manifest.json').read_text())


if __name__ == '__main__':
    main()
