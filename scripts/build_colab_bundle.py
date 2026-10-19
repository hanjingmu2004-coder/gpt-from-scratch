"""Bundle only source and prepared data, never environment/checkpoints/secrets."""
from pathlib import Path
import zipfile

root = Path(__file__).resolve().parents[1]
required = root / 'data/wikitext2-first/manifest.json'
if not required.exists():
    raise SystemExit('Run python -m minigpt.prepare_data first')
files = [root / name for name in ('README.md', 'requirements.txt', 'requirements-experiments.txt')]
for directory, pattern in (('minigpt', '*.py'), ('configs', '*.json'), ('tests', '*.py'),
                           ('docs', '*.md'), ('data/wikitext2-first', '*.jsonl'),
                           ('data/wikitext2-first', '*.json'), ('data/wikitext2-first', '*.md')):
    files.extend(sorted((root / directory).glob(pattern)))
output = root / 'colab-first-run.zip'
with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
    for path in files:
        archive.write(path, path.relative_to(root))
print(f'{output} ({output.stat().st_size:,} bytes, {len(files)} files)')
