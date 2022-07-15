import unittest

from dcan.loes_scoring.dsets import get_candidate_info_list


class TestDSets(unittest.TestCase):
    def test_get_candidate_info_list(self):
        candidate_info_list = get_candidate_info_list()
        self.assertIsNotNone(candidate_info_list)

if __name__ == '__main__':
    unittest.main()
