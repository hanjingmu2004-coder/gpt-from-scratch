import importlib.util
import unittest
import torch
from minigpt.model import Config, GPT
from minigpt.pretrained import convert_state


@unittest.skipUnless(importlib.util.find_spec('transformers'), 'Optional pretrained dependencies missing')
class PretrainedTests(unittest.TestCase):
    def test_reference_mapping_and_tying(self):
        from transformers import GPT2Config, GPT2LMHeadModel
        torch.set_num_threads(1)
        torch.manual_seed(5)
        reference = GPT2LMHeadModel(GPT2Config(vocab_size=32, n_positions=32,
            n_embd=16, n_head=2, n_layer=2, resid_pdrop=0., embd_pdrop=0.,
            attn_pdrop=0., attn_implementation='eager')).eval()
        cfg = Config(vocab_size=32, context_length=32, emb_dim=16, n_heads=2,
                     n_layers=2, dropout=0., qkv_bias=True, tie_embeddings=True)
        model = GPT(cfg).eval()
        model.load_state_dict(convert_state(reference.state_dict(), cfg))
        self.assertIs(model.head.weight, model.tok_emb.weight)
        ids = torch.randint(32, (2, 17))
        torch.testing.assert_close(model(ids), reference(ids).logits, atol=1e-6, rtol=1e-5)
        first, cache = model(ids[:, :8], use_cache=True)
        rest, _ = model(ids[:, 8:], past=cache, use_cache=True)
        torch.testing.assert_close(torch.cat((first, rest), 1), reference(ids).logits, atol=1e-6, rtol=1e-5)
