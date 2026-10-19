import unittest
import torch
from torch import nn
from minigpt.adapters import LowRank, attach, trainable_state, load_trainable
from minigpt.model import GPT, Config


class AdapterTests(unittest.TestCase):
    def test_merged_checkpoint_reloads_without_adapters(self):
        from minigpt.merge_adapter import merge
        cfg = Config(vocab_size=31, context_length=16, emb_dim=8,
                     n_heads=2, n_layers=2, dropout=0., tie_embeddings=True)
        for method in ('lora', 'rslora', 'dora'):
            torch.manual_seed(42)
            model = attach(GPT(cfg), method, rank=2).eval()
            with torch.no_grad():
                for module in model.modules():
                    if isinstance(module, LowRank):
                        module.b.normal_(std=.02)
                        if method == 'dora':
                            module.magnitude.mul_(1.1)
                ids = torch.randint(31, (2, 9))
                expected = model(ids)
                merge(model)
                restored = GPT(cfg).eval()
                restored.load_state_dict(model.state_dict(), strict=True)
                torch.testing.assert_close(restored(ids), expected, atol=2e-5, rtol=2e-5)
                self.assertFalse(any(isinstance(m, LowRank) for m in model.modules()))
                self.assertIs(restored.head.weight, restored.tok_emb.weight)

    def test_initial_identity_and_gradients(self):
        for method in ('lora', 'rslora', 'dora'):
            torch.manual_seed(123)
            base = nn.Linear(8, 12)
            x = torch.randn(2, 3, 8)
            expected = base(x).detach()
            adapter = LowRank(base, rank=2, method=method)
            torch.testing.assert_close(adapter(x), expected)
            adapter(x).square().sum().backward()
            self.assertIsNone(base.weight.grad)
            self.assertGreater(adapter.b.grad.abs().sum().item(), 0)

    def test_dora_effective_weight(self):
        adapter = LowRank(nn.Linear(8, 12), rank=2, method='dora')
        with torch.no_grad():
            adapter.b.normal_()
            adapter.magnitude.mul_(1.2)
        w = adapter.base.weight + adapter.scale * adapter.b @ adapter.a
        expected = w / w.norm(dim=1, keepdim=True) * adapter.magnitude[:, None]
        x = torch.randn(2, 8)
        torch.testing.assert_close(adapter(x), torch.nn.functional.linear(x, expected, adapter.base.bias))

    def test_state_roundtrip_freezing_and_scale(self):
        model = attach(GPT(Config(vocab_size=31, emb_dim=8, n_heads=2, n_layers=1)), rank=2)
        self.assertEqual(len(trainable_state(model)), 8)
        self.assertFalse(model.tok_emb.weight.requires_grad)
        state = trainable_state(model)
        with torch.no_grad():
            model.blocks[0].att.qkv.b.fill_(3)
        load_trainable(model, state)
        self.assertEqual(model.blocks[0].att.qkv.b.abs().sum(), 0)
        self.assertAlmostEqual(LowRank(nn.Linear(8, 8), 4, 16, 'rslora').scale, 8)

    def test_fixed_rank_rslora_is_identical_when_scale_matched(self):
        import copy
        rank, alpha = 4, 16.
        base = nn.Linear(8, 12)
        lora = LowRank(copy.deepcopy(base), rank, alpha, 'lora')
        rs = LowRank(copy.deepcopy(base), rank, alpha / rank ** .5, 'rslora')
        with torch.no_grad():
            lora.b.normal_()
            rs.a.copy_(lora.a)
            rs.b.copy_(lora.b)
        x = torch.randn(2, 8)
        torch.testing.assert_close(lora(x), rs(x), rtol=0, atol=0)
        lora(x).sum().backward()
        rs(x).sum().backward()
        torch.testing.assert_close(lora.a.grad, rs.a.grad, rtol=0, atol=0)
        torch.testing.assert_close(lora.b.grad, rs.b.grad, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
