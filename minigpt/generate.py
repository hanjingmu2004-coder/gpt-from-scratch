import torch


@torch.no_grad()
def generate(model, ids, max_new_tokens, use_cache=True, temperature=0.0,
             top_k=None, eos_id=None):
    if max_new_tokens < 0 or temperature < 0:
        raise ValueError("Invalid generation settings")
    model.eval()
    cache = None
    finished = torch.zeros(ids.size(0), dtype=torch.bool, device=ids.device)
    for _ in range(max_new_tokens):
        context = ids[:, -model.cfg.context_length:]
        # Learned absolute positions require rebuilding after a sliding-window shift.
        if use_cache:
            if cache is None or cache[0][0].size(2) >= model.cfg.context_length:
                logits, cache = model(context, use_cache=True)
            else:
                logits, cache = model(ids[:, -1:], past=cache, use_cache=True)
        else:
            logits = model(context)
        logits = logits[:, -1, :]
        if top_k is not None:
            if not 1 <= top_k <= logits.size(-1):
                raise ValueError("top_k outside vocabulary")
            cutoff = logits.topk(top_k).values[:, -1:]
            logits = logits.masked_fill(logits < cutoff, float('-inf'))
        nxt = (torch.multinomial((logits / temperature).softmax(-1), 1)
               if temperature > 0 else logits.argmax(-1, keepdim=True))
        if eos_id is not None:
            nxt = torch.where(finished[:, None], eos_id, nxt)
            finished |= nxt[:, 0] == eos_id
        ids = torch.cat((ids, nxt), dim=1)
        if eos_id is not None and finished.all():
            break
    return ids


def main():
    import argparse
    import tiktoken
    from .model import Config, GPT
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--prompt', required=True)
    p.add_argument('--instruction', action='store_true', help='Apply the SFT prompt template')
    p.add_argument('--max-new-tokens', type=int, default=64)
    p.add_argument('--temperature', type=float, default=0.)
    p.add_argument('--top-k', type=int)
    p.add_argument('--no-cache', action='store_true')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if state['task'] == 'classification':
        p.error('Generation requires an LM or SFT checkpoint')
    model = GPT(Config(**state['config'])).to(args.device)
    model.load_state_dict(state['model'])
    tok = tiktoken.get_encoding('gpt2')
    text = args.prompt
    if args.instruction:
        from .data import prompt
        text = prompt({'instruction': text})
    tokens = tok.encode(text)
    if not tokens:
        p.error('Prompt cannot be empty')
    ids = torch.tensor([tokens], device=args.device)
    result = generate(model, ids, args.max_new_tokens, use_cache=not args.no_cache,
                      temperature=args.temperature, top_k=args.top_k, eos_id=tok.eot_token)
    print(tok.decode(result[0, len(tokens):].tolist()))


if __name__ == '__main__':
    main()
