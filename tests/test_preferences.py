import unittest
import torch
from minigpt.preference_objectives import dpo, group_advantages, exact_kl, clipped_policy_loss


class PreferenceTests(unittest.TestCase):
    def test_dpo_reference_and_improving_preference(self):
        ref = torch.tensor([[0., 0.]]).log_softmax(-1)
        self.assertAlmostEqual(dpo(ref, ref, torch.tensor([1])).item(), __import__('math').log(2), places=6)
        improved = torch.tensor([[-1., 1.]]).log_softmax(-1)
        self.assertLess(dpo(improved, ref, torch.tensor([1])), dpo(ref, ref, torch.tensor([1])))

    def test_group_baseline_and_degenerate_groups(self):
        rewards = torch.tensor([[1., 1., 1., 1.], [0., 1., 0., 1.]])
        advantages = group_advantages(rewards)
        torch.testing.assert_close(advantages[0], torch.zeros(4))
        torch.testing.assert_close(advantages[1], torch.tensor([-1., 1., -1., 1.]))

    def test_kl_and_clipping(self):
        ref = torch.tensor([[0., 0.]]).log_softmax(-1)
        self.assertEqual(exact_kl(ref, ref).item(), 0)
        current = torch.tensor([[.9, .1]]).log()
        actions = torch.tensor([[0]])
        old = torch.tensor([[.5]]).log()
        loss = clipped_policy_loss(current, old, actions, torch.ones(1, 1))
        self.assertAlmostEqual(loss.item(), -1.2, places=6)
        self.assertGreater(exact_kl(current, ref).item(), 0)


if __name__ == '__main__':
    unittest.main()
