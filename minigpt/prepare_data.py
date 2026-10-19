"""Pinned WikiText-2 download, article reconstruction and auditable subsets."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import re
import urllib.request

REVISION = '6231e49f19a707241d6b84d9cff60a3a86b85a85'
BASE = f'https://huggingface.co/datasets/Salesforce/wikitext/resolve/{REVISION}'
SOURCE = 'https://huggingface.co/datasets/Salesforce/wikitext'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def articles(lines):
    """Top-level '= Title =' starts an article; '== Section ==' does not."""
    result, current = [], []
    for line in lines:
        if re.fullmatch(r'= [^=].* =', line.strip()) and current:
            result.append(''.join(current).strip())
            current = []
        current.append(line if line.endswith('\n') else line + '\n')
    if current:
        result.append(''.join(current).strip())
    return [text for text in result if text]


def normalized(text):
    return ' '.join(text.casefold().split())


def deduplicate(splits):
    seen, output, dropped = set(), {}, {}
    # Preserve evaluation membership when a duplicate also occurs in training.
    for split in ('test', 'val', 'train'):
        output[split], dropped[split] = [], 0
        for text in splits[split]:
            key = hashlib.sha256(normalized(text).encode()).hexdigest()
            if key in seen:
                dropped[split] += 1
            else:
                seen.add(key)
                output[split].append(text)
    return output, dropped


def download(url, path):
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={'User-Agent': 'minigpt-data-preparation/1.0'})
    temporary = path.with_suffix(path.suffix + '.part')
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open('wb') as f:
        while block := response.read(1024 * 1024):
            f.write(block)
    temporary.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', default='data/wikitext2-first')
    p.add_argument('--cache', default='data/raw/wikitext2')
    p.add_argument('--train-docs', type=int, default=128, help='0 uses all documents')
    p.add_argument('--val-docs', type=int, default=16)
    p.add_argument('--test-docs', type=int, default=16)
    p.add_argument('--seed', type=int, default=123)
    args = p.parse_args()
    if min(args.train_docs, args.val_docs, args.test_docs) < 0:
        p.error('Document limits must be nonnegative')
    out, cache = Path(args.out), Path(args.cache)
    spec = dict(revision=REVISION, seed=args.seed, train_docs=args.train_docs,
                val_docs=args.val_docs, test_docs=args.test_docs, preprocessing_version=1)
    manifest_path = out / 'manifest.json'
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding='utf-8'))
        if existing['spec'] != spec or any(sha256(out / (s + '.jsonl')) != v['sha256']
                                           for s, v in existing['splits'].items()):
            raise ValueError('Existing data differ; use a new --out directory')
        print(json.dumps(existing, ensure_ascii=False, indent=2))
        return
    if out.exists() and any(out.iterdir()):
        raise ValueError('Output must be empty or contain a verified manifest')
    import pyarrow.parquet as pq
    import tiktoken
    os.environ.setdefault('TIKTOKEN_CACHE_DIR', str(cache / 'tokenizer'))
    tokenizer = tiktoken.get_encoding('gpt2')
    splits, sources = {}, {}
    for name, upstream in (('train', 'train'), ('val', 'validation'), ('test', 'test')):
        relative = f'wikitext-2-raw-v1/{upstream}-00000-of-00001.parquet'
        path = cache / Path(relative).name
        print(f'Downloading/reading {upstream}', flush=True)
        download(f'{BASE}/{relative}', path)
        rows = pq.read_table(path, columns=['text'])['text'].to_pylist()
        splits[name] = articles(rows)
        sources[name] = {'url': f'{BASE}/{relative}', 'sha256': sha256(path),
                         'raw_rows': len(rows), 'articles_before_dedup': len(splits[name])}
    card = cache / 'SOURCE_README.md'
    download(f'{BASE}/README.md', card)
    splits, dropped = deduplicate(splits)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {'spec': spec, 'source': SOURCE, 'source_card_sha256': sha256(card),
                'license': 'cc-by-sa-3.0 and gfdl (dataset card metadata; see SOURCE_README.md)',
                'preprocessing': 'Official splits; reconstruct top-level articles; whitespace/case-normalized exact article dedup; deterministic document subset; no near-duplicate removal',
                'tokenizer': 'tiktoken gpt2', 'splits': {}, 'sources': sources}
    for index, name in enumerate(('train', 'val', 'test')):
        texts = splits[name].copy()
        random.Random(args.seed + index).shuffle(texts)
        limit = getattr(args, name + '_docs')
        if limit:
            texts = texts[:limit]
        if not texts:
            raise ValueError(f'Empty split: {name}')
        path = out / f'{name}.jsonl'
        with path.open('w', encoding='utf-8', newline='\n') as f:
            for text in texts:
                f.write(json.dumps({'text': text}, ensure_ascii=False) + '\n')
        lengths = [len(tokenizer.encode(text)) + 1 for text in texts]
        manifest['splits'][name] = {'documents': len(texts), 'tokens_including_eos': sum(lengths),
                                   'dropped_duplicate_articles': dropped[name], 'sha256': sha256(path)}
    (out / 'SOURCE_README.md').write_bytes(card.read_bytes())
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
