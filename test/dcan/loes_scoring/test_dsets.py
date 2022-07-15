import unittest

from dcan.loes_scoring.dsets import get_candidate_info_list, LoesScoreMRIs


class TestDSets(unittest.TestCase):
    def test_get_candidate_info_list(self):
        candidate_info_list = get_candidate_info_list()
        self.assertIsNotNone(candidate_info_list)

    def test_LoesScoreMRIs_init(self):
        loes_score_mris = LoesScoreMRIs('5772LAVA_20180828')
        self.assertIsNotNone(loes_score_mris)
        raw_candidate = loes_score_mris.get_raw_candidate()
        self.assertIsNotNone(raw_candidate[0])
        self.assertIsNotNone(raw_candidate[0])


if __name__ == '__main__':
    unittest.main()
