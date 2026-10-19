"""Embed explicit source/public-data allowlist for the VS Code Colab extension."""
import base64
import io
import json
from pathlib import Path
import zipfile

root = Path(__file__).resolve().parents[1]
files = list((root / 'minigpt').glob('*.py')) + list((root / 'tests').glob('*.py'))
files += [root / n for n in ('requirements.txt', 'requirements-experiments.txt',
                            'requirements-pretrained.txt', 'requirements-comparison.txt')]
for folder in ('data/instructions', 'data/sentiment-control'):
    files += [p for p in (root / folder).iterdir() if p.suffix in ('.jsonl', '.json', '.txt')]
buffer = io.BytesIO()
with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as z:
    for p in sorted(files):
        z.write(p, p.relative_to(root).as_posix())
payload = base64.b64encode(buffer.getvalue()).decode()

setup = '''from pathlib import Path
import base64, io, zipfile, os, sys, subprocess, json, hashlib, datetime, gc, uuid
PAYLOAD = "PAYLOAD_VALUE"
project = Path('/content/minigpt-method-comparison')
project.mkdir(exist_ok=True)
with zipfile.ZipFile(io.BytesIO(base64.b64decode(PAYLOAD))) as z:
    z.extractall(project)
(project / 'source-and-data.zip').write_bytes(base64.b64decode(PAYLOAD))
os.chdir(project)
os.environ['OMP_NUM_THREADS']='2'
os.environ['MKL_NUM_THREADS']='2'
subprocess.run([sys.executable,'-m','pip','install','-q','bitsandbytes==0.50.0','tiktoken','transformers==4.57.6','safetensors'],check=True)
import torch
assert torch.cuda.is_available(), 'Select a Colab GPU kernel'
print('GPU:',torch.cuda.get_device_name(), 'torch:',torch.__version__)
for name in ('selected',):
    globals().pop(name,None)
gc.collect()
torch.cuda.empty_cache()
pretrained=Path('/content/minigpt-gpt2-sft/pretrained/model.pt')
if not pretrained.exists():
    subprocess.run([sys.executable,'-m','minigpt.pretrained','--out',str(project/'pretrained')],check=True)
    pretrained=project/'pretrained/model.pt'
print('Base checkpoint SHA256:',hashlib.sha256(pretrained.read_bytes()).hexdigest())
comparison_run=project/'runs'/('comparison-'+datetime.datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:6])
comparison_run.mkdir(parents=True,exist_ok=True)
(comparison_run/'source-and-data.zip').write_bytes((project/'source-and-data.zip').read_bytes())
(comparison_run/'environment.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
(comparison_run/'base_provenance.json').write_text(json.dumps({'sha256':hashlib.sha256(pretrained.read_bytes()).hexdigest(),'path':str(pretrained)},indent=2))
print('RUN:',comparison_run)
def run_job(module, arguments, dest, completed):
    if (dest/completed).exists():
        print('Already completed:',dest,flush=True)
        return
    actual=dest
    attempt=1
    while actual.exists():
        actual=dest.with_name(dest.name+f'-retry{attempt}')
        if (actual/completed).exists():
            print('Already completed:',actual,flush=True)
            return
        attempt+=1
    actual.parent.mkdir(parents=True,exist_ok=True)
    cmd=[sys.executable,'-u','-m',module,'--init',str(pretrained),'--out',str(actual)]+arguments
    print('START',actual,flush=True)
    with (actual.parent/(actual.name+'.log')).open('w') as log:
        process=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        try:
            for line in process.stdout:
                log.write(line);log.flush();print(line,end='',flush=True)
            code=process.wait()
        except BaseException:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=15)
            raise
        if code:
            raise RuntimeError(f'{actual.name} exited {code}; see saved log')
'''.replace('PAYLOAD_VALUE', payload)
sft = '''# Fixed dataset/order budget: 2 epochs, batch=4, three seeds.
for method in ('full','lora','rslora','dora','qlora'):
    for seed in (123,456,789):
        run_job('minigpt.compare_methods',['--method',method,'--seed',str(seed)],
                comparison_run/'sft'/f'{method}-{seed}','result.json')
print('SFT MATRIX COMPLETE',flush=True)
'''
rl = '''# Categorical single-decision task; this is not a long-text RLHF benchmark.
for seed in (123,456,789):
    run_job('minigpt.compare_preferences',['--seed',str(seed)],
            comparison_run/'preferences'/f'seed-{seed}','COMPLETE.json')
print('PREFERENCE MATRIX COMPLETE',flush=True)
'''
export = '''import shutil
archive=project/'comparison-results.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for path in comparison_run.rglob('*'):
        if path.is_file():
            # Full SFT weights stay remote (about 500MB each); adapters are included.
            if path.name=='selected-trainable.pt' and path.parent.name.startswith('full-'):
                continue
            z.write(path,path.relative_to(comparison_run))
print('DOWNLOAD:',archive,'bytes:',archive.stat().st_size)
print('SHA256:',hashlib.sha256(archive.read_bytes()).hexdigest())
print('All result records:')
for path in comparison_run.rglob('result.json'):
    print(path.parent.name,json.loads(path.read_text()))
for path in comparison_run.rglob('results.json'):
    for row in json.loads(path.read_text()):
        print(path.parent.name,row['method'],'test accuracy',row['test']['accuracy'])
'''

def cell(source, kind='code'):
    result = dict(cell_type=kind, metadata={}, source=source.splitlines(keepends=True))
    if kind == 'code':
        result.update(execution_count=None, outputs=[])
    return result

nb = dict(cells=[cell('# LoRA 与偏好/奖励优化：可核验的对照实验\n\n'
                     'GPT-2 124M；3 个 seed。SFT 比较 full/LoRA/rsLoRA/DoRA/NF4-LoRA。'
                     'SST-2 二标签单步任务比较继续 SFT/DPO/PPO/GRPO；不能外推为长文本推理排名。\n', 'markdown'),
                 cell(setup), cell(sft), cell(rl), cell(export)],
          metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
                    'language_info':{'name':'python'}}, nbformat=4, nbformat_minor=5)
path = root / 'VSCode_Colab_Method_Comparison.ipynb'
path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
print(path, 'payload bytes', len(buffer.getvalue()))
