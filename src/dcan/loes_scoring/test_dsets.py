import unittest

from dcan.loes_scoring.dsets import get_candidate_info_list, LoesScoreMRIs, get_loes_score_mris, \
    get_mri_raw_candidate, LoesScoreDataset


class TestDSets(unittest.TestCase):
    def test_get_candidate_info_list(self):
        candidate_info_list = get_candidate_info_list()
        self.assertIsNotNone(candidate_info_list)

    def test_LoesScoreMRIs_init(self):
        loes_score_mris = LoesScoreMRIs('7295TASU_20190220')
        self.assertIsNotNone(loes_score_mris)
        raw_candidate = loes_score_mris.get_raw_candidate()
        self.assertIsNotNone(raw_candidate[0])
        self.assertIsNotNone(raw_candidate[0])

    def test_get_loes_score_mris(self):
        loes_score_mris = get_loes_score_mris('5772LAVA_20180828')
        self.assertIsNotNone(loes_score_mris)

    def test_get_mri_raw_candidate(self):
        mri_raw_candidate = get_mri_raw_candidate('5772LAVA_20180828')
        self.assertIsNotNone(mri_raw_candidate[0])
        self.assertIsNotNone(mri_raw_candidate[0])

    def test_loes_score_dataset_init(self):
        loes_score_dataset = LoesScoreDataset(val_stride=5)
        self.assertIsNotNone(loes_score_dataset)
        length = loes_score_dataset.__len__()
        self.assertEqual(4, length)
        for i in range(length):
            item = loes_score_dataset.__getitem__(i)
            self.assertIsNotNone(item)


if __name__ == '__main__':
    unittest.main()
