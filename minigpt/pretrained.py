"""Convert the public OpenAI GPT-2 checkpoint distributed by Hugging Face."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import torch
from .model import Config, GPT

MODEL_ID = 'openai-community/gpt2'
REVISION = '607a30d783dfa663caf39e06633721c8d4cfcd7e'


def convert_state(source, cfg):
    target = {'tok_emb.weight': source['transformer.wte.weight'],
              'pos_emb.weight': source['transformer.wpe.weight'],
              'norm.weight': source['transformer.ln_f.weight'],
              'norm.bias': source['transformer.ln_f.bias'],
              'head.weight': source['transformer.wte.weight']}
    for i in range(cfg.n_layers):
        a, b = f'transformer.h.{i}.', f'blocks.{i}.'
        for norm in (1, 2):
            for kind in ('weight', 'bias'):
                target[f'{b}norm{norm}.{kind}'] = source[f'{a}ln_{norm}.{kind}']
        for old, new in (('attn.c_attn', 'att.qkv'), ('attn.c_proj', 'att.out'),
                         ('mlp.c_fc', 'ff.0'), ('mlp.c_proj', 'ff.2')):
            target[b + new + '.weight'] = source[a + old + '.weight'].T.contiguous()
            target[b + new + '.bias'] = source[a + old + '.bias']
    return target


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', default='runs/gpt2-pretrained')
    p.add_argument('--device', default='cpu')
    args = p.parse_args()
    from transformers import GPT2LMHeadModel
    reference = GPT2LMHeadModel.from_pretrained(MODEL_ID, revision=REVISION,
                attn_implementation='eager', use_safetensors=True).to(args.device).eval()
    cfg = Config(vocab_size=50257, context_length=1024, emb_dim=768,
                 n_heads=12, n_layers=12, dropout=0., qkv_bias=True, tie_embeddings=True)
    converted = GPT(cfg).to(args.device).eval()
    converted.load_state_dict(convert_state(reference.state_dict(), cfg), strict=True)
    errors = []
    with torch.no_grad():
        # Reference equivalence is checked on multiple sequences, before training.
        for length in (8, 31):
            torch.manual_seed(123 + length)
            ids = torch.randint(cfg.vocab_size, (1, length), device=args.device)
            expected = reference(ids).logits
            actual = converted(ids)
            torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-4)
            errors.append(float((actual - expected).abs().max()))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'model.pt').exists():
        raise ValueError('Output exists; choose another directory')
    state = dict(task='lm', config=asdict(cfg), model={k:v.cpu() for k,v in converted.state_dict().items()})
    torch.save(state, out / 'model.pt')
    provenance = dict(model=MODEL_ID, revision=REVISION, original_project='https://github.com/openai/gpt-2',
                      distribution='Hugging Face safetensors serialization of OpenAI GPT-2',
                      parameters=sum(p.numel() for p in converted.parameters()),
                      reference_max_abs_errors=errors, reference_tolerance=dict(atol=2e-4, rtol=2e-4),
                      checkpoint_sha256=hashlib.sha256((out/'model.pt').read_bytes()).hexdigest())
    (out / 'provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    print(json.dumps(provenance, indent=2), flush=True)


if __name__ == '__main__':
    main()
