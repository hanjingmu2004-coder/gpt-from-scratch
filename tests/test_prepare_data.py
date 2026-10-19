import unittest
from minigpt.prepare_data import articles, deduplicate


class DataPreparationTests(unittest.TestCase):
    def test_article_boundary(self):
        result = articles(['\n', ' = Alpha = \n', 'body\n', ' == Section == \n',
                           'more\n', ' = Beta = \n', 'second\n'])
        self.assertEqual(len(result), 2)
        self.assertIn('Section', result[0])
        self.assertTrue(result[1].startswith('= Beta ='))

    def test_dedup_preserves_evaluation(self):
        splits, dropped = deduplicate({'train': ['same TEXT', 'new', 'new'],
                                       'val': ['other'], 'test': [' same   text ']})
        self.assertEqual(splits['train'], ['new'])
        self.assertEqual(splits['test'], [' same   text '])
        self.assertEqual(dropped['train'], 2)


if __name__ == '__main__':
    unittest.main()
