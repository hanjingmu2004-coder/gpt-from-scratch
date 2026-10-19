import unittest
import torch
from minigpt.model import Config, GPT, Classifier
from minigpt.data import collate, encode_rows
from minigpt.generate import generate
from minigpt.train import evaluate


class FakeTokenizer:
    eot_token = 31
    n_vocab = 32

    def encode(self, text):
        return [ord(c) % 30 for c in text]


class CoreTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(5)
        torch.set_num_threads(1)
        self.model = GPT(Config(vocab_size=32, context_length=16, emb_dim=16,
                                n_heads=2, n_layers=2, dropout=0.)).eval()
        self.ids = torch.randint(0, 32, (2, 8))

    def test_causal(self):
        other = self.ids.clone()
        other[:, 4:] = (other[:, 4:] + 1) % 32
        torch.testing.assert_close(self.model(self.ids)[:, :4], self.model(other)[:, :4])

    def test_cache_logits_and_window(self):
        full = self.model(self.ids)
        prefix, cache = self.model(self.ids[:, :3], use_cache=True)
        pieces = [prefix]
        for i in range(3, 8):
            logits, cache = self.model(self.ids[:, i:i+1], cache, True)
            pieces.append(logits)
        torch.testing.assert_close(full, torch.cat(pieces, 1), atol=1e-6, rtol=1e-5)
        self.assertTrue(torch.equal(generate(self.model, self.ids, 14),
                                    generate(self.model, self.ids, 14, use_cache=False)))

    def test_sft_mask_and_eos(self):
        row = {'instruction': 'a', 'output': 'xy'}
        sample = encode_rows([row], FakeTokenizer(), 'sft', 128)[0]
        active = [v for v in sample[1] if v != -100]
        self.assertEqual(active, FakeTokenizer().encode('xy') + [31])
        _, y, _ = collate([sample, ([1], [31])], pad=31)
        self.assertTrue((y[1, 1:] == -100).all())
        with self.assertRaises(ValueError):
            encode_rows([row], FakeTokenizer(), 'sft', 2)

    def test_weighted_eval(self):
        samples = [([1, 2, 3], [2, 3, 4]), ([5], [6])]
        together = evaluate(self.model, [collate(samples, 31)], 'cpu')
        apart = evaluate(self.model, [collate([s], 31) for s in samples], 'cpu')
        self.assertAlmostEqual(together['loss'], apart['loss'], places=6)

    def test_classification_padding_and_modes(self):
        classifier = Classifier(self.model, 2).eval()
        short = classifier(self.ids[:, :4], torch.tensor([4, 4]))
        padded = classifier(self.ids, torch.tensor([4, 4]))
        torch.testing.assert_close(short, padded)
        classifier.configure('head')
        self.assertFalse(any(p.requires_grad for p in classifier.backbone.parameters()))
        classifier.configure('last')
        self.assertTrue(all(p.requires_grad for p in classifier.backbone.blocks[-1].parameters()))

    def test_roundtrip_and_training(self):
        import io
        stream = io.BytesIO()
        torch.save(self.model.state_dict(), stream)
        stream.seek(0)
        clone = GPT(self.model.cfg)
        clone.load_state_dict(torch.load(stream, weights_only=True))
        torch.testing.assert_close(self.model(self.ids), clone(self.ids))
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=.02)
        losses = []
        for _ in range(15):
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(self.model(self.ids).flatten(0, 1), self.ids.flatten())
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
        self.assertLess(losses[-1], losses[0])

    def test_training_entrypoints_and_resume(self):
        import json
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace
        from unittest.mock import patch
        from minigpt.train import main
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            config = root / 'config.json'
            config.write_text(json.dumps(dict(vocab_size=32, context_length=128,
                emb_dim=8, n_heads=2, n_layers=1, dropout=0.)))
            def run(task, name, extra=()):
                paths = []
                for i, split in enumerate(('train', 'val', 'test')):
                    row = ({'instruction': f'task{i}', 'output': 'yes'} if task == 'sft'
                           else {'text': f'hello{i}', 'label': i % 2})
                    path = root / f'{task}-{split}.jsonl'
                    path.write_text(json.dumps(row) + '\n')
                    paths += ['--' + split, str(path)]
                argv = ['train', '--task', task, '--config', str(config),
                        '--out', str(root / name), '--epochs', '1', '--device', 'cpu',
                        *paths, *extra]
                with patch('sys.argv', argv), patch.dict('sys.modules',
                    {'tiktoken': SimpleNamespace(get_encoding=lambda _: FakeTokenizer())}):
                    main()
                self.assertTrue((root / name / 'test_metrics.json').exists())
            run('lm', 'lm')
            run('lm', 'lm', ['--resume', str(root / 'lm/last.pt'), '--epochs', '2'])
            for mode in ('head', 'last', 'full'):
                run('classification', mode, ['--init', str(root / 'lm/best.pt'), '--mode', mode])
            run('sft', 'sft', ['--init', str(root / 'lm/best.pt')])


if __name__ == '__main__':
    unittest.main()
